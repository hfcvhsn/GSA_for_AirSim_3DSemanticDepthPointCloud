"""mk1 selected-camera and heading-frame regressions; segmentation is a CPU fixture."""
import json
import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation
from src.config import load_config, ROOT
from src.pipeline import GSAActionPipeline

FOUR=('FrontCamera','LeftCamera','RightCamera','DownCamera')


def test_four_views_without_rear_and_gravity_alignment(tmp_path,monkeypatch):
    source=tmp_path/'sample'
    for camera in FOUR:
        name=camera.lower();(source/name).mkdir(parents=True);(source/(name+'_depth')).mkdir()
        Image.fromarray(np.full((16,16,3),100,np.uint8)).save(source/name/'000000.png')
        Image.fromarray(np.full((16,16),20,np.uint8)).save(source/(name+'_depth')/'000000.png')
    (source/'log').mkdir()
    rotation=Rotation.from_euler('xyz',[18,-12,37],degrees=True)
    (source/'log/000000.json').write_text(json.dumps(dict(sensors=dict(state=dict(position=[3,4,-10],orientation=rotation.as_quat().tolist())))))
    class Segmenter:
        def __init__(self,config,classes,device):self.classes=classes;self.settings=config['segmentation'];self.device=device
        def segment(self,path,max_detections):
            rgb=path if isinstance(path,np.ndarray) else np.array(Image.open(path));shape=rgb.shape[:2]
            return rgb,np.ones(shape,np.uint16),np.ones(shape,np.uint16),np.ones(shape,np.float32),[]
    monkeypatch.setattr('src.pipeline.GroundedMobileSAM',Segmenter)
    config=load_config(ROOT/'config/resources.example.json');config['cloud'].update(stride=1,voxel_size=0)
    pipeline=GSAActionPipeline(config,device='cpu',classes=['tree'],cameras=FOUR,gravity_aligned=True)
    output=tmp_path/'out';rows=pipeline.run(source,output=output)
    assert len(rows)==4 and pipeline.last_cloud is None
    capture=json.loads((output/'capture/meta.json').read_text())
    assert set(capture['cameras'])==set(FOUR)
    assert not (output/'capture/RearCamera').exists() and not (output/'cloud').exists()
    from src.geometry import backproject, body_to_action_rotation
    from src.actions import action_membership, pool_action_samples, format_action_rows
    observations, metadata=pipeline._load(source, None, None)
    T=body_to_action_rotation(metadata['vehicle_pose_world_ned'])
    all_distance, all_labels, all_inside=[], [], []
    for obs in observations:
        xyz,v,u=backproject(obs.depth_perspective_m,obs.metadata,metadata['vehicle_pose_world_ned'])
        xyz=xyz@T;inside=action_membership(xyz);selected=inside.any(axis=1)
        all_distance.extend(np.linalg.norm(xyz[selected],axis=1));all_labels.extend([1]*int(selected.sum()));all_inside.extend(inside[selected])
    features=pool_action_samples(all_distance,all_labels,np.asarray(all_inside),2)
    expected=format_action_rows(features,['unknown','tree'])
    assert [r['semantic_id'] for r in rows]==[r['semantic_id'] for r in expected]
    np.testing.assert_allclose([r['depth_m'] for r in rows],[r['depth_m'] for r in expected],atol=2e-6)
    assert pipeline.last_report['action_frame']=='heading_FRD'


def test_persistent_stream_reuses_one_pipeline(monkeypatch,capsys):
    import io
    import scripts.run_pipeline as cli
    instances=[]
    class Pipeline:
        def __init__(self,*args,**kwargs):
            self.cameras=tuple(kwargs['cameras']);self._model=None;instances.append(self)
        def run(self,*args,**kwargs):
            self._model=True;self.last_report=dict(timings_seconds=dict(segmentation=.1))
            return [dict(action='front',depth_m=None,semantic_id=-1,semantic_name='no_return')]
    monkeypatch.setattr(cli,'GSAActionPipeline',Pipeline)
    monkeypatch.setattr('sys.stdin',io.StringIO('\n'.join(json.dumps(dict(id=i,input='/unused')) for i in range(2))+'\n'))
    assert cli.main(['--stream','--no-ram-tags','--cameras',*FOUR,'--gravity-aligned'])==0
    rows=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[0]['ready'] and len(instances)==1
    assert [r['id'] for r in rows[1:]]==[0,1]
    assert rows[1]['cold_start'] is True and rows[2]['cold_start'] is False


def test_default_stream_returns_action_rows_and_ram_tags(monkeypatch, capsys):
    import scripts.run_pipeline as cli
    actions = [dict(action=name, depth_m=None, semantic_id=-1, semantic_name='no_return')
               for name in ('front', 'L30', 'R30', 'DOWN')]
    tags = {'front': {'tags': ['road'], 'tags_zh': ['道路']}, 'down': {'tags': [], 'tags_zh': []}}
    class Pipeline:
        cameras = ('FrontCamera', 'DownCamera')
        _model = None
        last_report = {}
        def __init__(self, *args, **kwargs):
            pass
        def run(self, *args, **kwargs):
            assert kwargs['ram_tags'] is True
            return {'actions': actions, 'ram_tags': tags}
    monkeypatch.setattr(cli, 'GSAActionPipeline', Pipeline)
    monkeypatch.setattr('sys.stdin', io.StringIO('{"id":7,"input":"unused"}\n'))
    assert cli.main(['--stream']) == 0
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages[0]['cameras'] == ['FrontCamera', 'DownCamera']
    assert messages[1]['rows'] == actions and messages[1]['ram_tags'] == tags


@pytest.mark.parametrize('capture_failure', [False, True])
def test_live_stream_real_rpc_constructor_and_four_view_capture(tmp_path,monkeypatch,capsys,capture_failure):
    """Keep installed AirSim/RPC constructors real; replace RPC replies and ML only.

    The former raw MultirotorClient call fails this regression with the installed
    RPC fork before any fixture reply is used (unexpected pack_encoding).
    """
    import scripts.run_pipeline as cli
    import src.openuav.client as rpc
    from src.openuav.dataset import pose_dict
    real_client = rpc.OpenUAVClient
    layout = json.loads((ROOT/'config/openuav_cameras.json').read_text())['cameras']
    calls,closed = [],[]
    def airsim_pose(position,rotation):
        pose=pose_dict(position,rotation)
        return dict(position={k+'_val':v for k,v in pose['position'].items()},
                    orientation={k+'_val':v for k,v in pose['orientation_wxyz'].items()})
    body=airsim_pose([0,0,-10],Rotation.identity())
    poses={name:airsim_pose(np.array([0,0,-10])+spec['position'],
                           Rotation.from_euler('xyz',spec['rpy_deg'],degrees=True))
           for name,spec in layout.items()}
    def reply(method,*args):
        calls.append((method,args))
        if method=='simIsPaused':return True
        if method=='simGetVehiclePose':return body
        if method=='simGetCameraInfo':return dict(fov=90.,pose=poses[args[0]],proj_mat=dict(matrix=np.eye(4).tolist()))
        if method=='simGetImages':
            if capture_failure:raise RuntimeError('fixture capture failure')
            return [dict(width=16,height=16,camera_position=poses[r.camera_name]['position'],
                         camera_orientation=poses[r.camera_name]['orientation'],time_stamp=100,
                         image_data_uint8=bytes([0,0,255]*256),image_data_float=np.linspace(6,8,256).tolist())
                    for r in args[0]]
        raise AssertionError(method)
    def connect(**kwargs):
        client=real_client(**kwargs)  # Actual installed msgpackrpc.Client signature.
        original_close=client.client.close
        def close():closed.append(True);original_close()
        client.client.call=reply;client.client.close=close
        return client
    class Segmenter:
        def __init__(self,config,classes,device):self.classes=classes;self.settings=config['segmentation'];self.device=device
        def segment(self,path,max_detections):
            rgb=path if isinstance(path,np.ndarray) else np.asarray(Image.open(path));shape=rgb.shape[:2]
            return rgb,np.ones(shape,np.uint16),np.ones(shape,np.uint16),np.ones(shape,np.float32),[]
    monkeypatch.setattr(rpc,'OpenUAVClient',connect)
    monkeypatch.setattr('src.pipeline.GroundedMobileSAM',Segmenter)
    capture=tmp_path/'capture'
    request=dict(id='live:0',input=str(capture),live=dict(ip='127.0.0.1',port=41451))
    monkeypatch.setattr('sys.stdin',io.StringIO(json.dumps(request)+'\n'))
    argv=['--stream','--no-ram-tags','--device','cpu','--config',str(ROOT/'config/resources.example.json'),
          '--cameras',*FOUR,'--gravity-aligned']
    if capture_failure:
        with pytest.raises(SystemExit) as error:cli.main(argv)
        assert error.value.code==1
    else:
        assert cli.main(argv)==0
        output=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert output[0]['ready'] and output[1]['id']=='live:0'
        assert len(output[1]['rows'])==4
        report=output[1]['report']
        assert report['status']=='completed' and report['depth_quality']=='metric_float'
        assert report['action_frame']=='heading_FRD' and tuple(report['cameras'])==FOUR
        metadata=json.loads((capture/'meta.json').read_text())
        assert tuple(metadata['cameras'])==FOUR and metadata['pose_synchronized_by_pause']
        assert not (capture/'RearCamera').exists()
        assert len(list(capture.glob('*/rgb.png')))==len(list(capture.glob('*/depth_perspective_m.npy')))==4
    requests=next(args[0] for method,args in calls if method=='simGetImages')
    assert [(r.camera_name,r.image_type) for r in requests]==[(name,t) for name in FOUR for t in (0,2)]
    assert closed==[True]
