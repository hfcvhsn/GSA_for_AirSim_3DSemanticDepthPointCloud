"""Four perception cones, scalar pixel pooling and legacy cloud-query parity."""
import json
import numpy as np
import pytest
from src.actions import (ACTION_NAMES, DIRECTION_FRD, ELEVATION_DEG, YAW_DEG, action_membership,
                         get_action_depth_semantics, kde_depth_peak, query_actions)


def rays(yaw, elevation, distance):
    yaw, elevation = np.radians(yaw), np.radians(elevation)
    return np.column_stack((np.cos(elevation)*np.cos(yaw), np.cos(elevation)*np.sin(yaw),
                            -np.sin(elevation)))*np.asarray(distance)[:, None]


def test_four_outputs_and_missing_vs_unknown():
    xyz = DIRECTION_FRD * np.array([1., 16., 3., 4.])[:, None]
    rows = get_action_depth_semantics(xyz, [0, 2, 3, 4], ['unknown', 'a', 'b', 'c', 'd'])
    assert [r['action'] for r in rows] == ['front', 'L30', 'R30', 'DOWN']
    assert [r['depth_m'] for r in rows] == [1., None, 3., 4.]
    assert [r['semantic_id'] for r in rows] == [0, -1, 3, 4]
    assert rows[0]['semantic_name'] == 'unknown' and rows[1]['semantic_name'] == 'no_return'
    assert json.loads(json.dumps(rows, allow_nan=False)) == rows


def test_downward_cap_is_independent_of_yaw_and_up_is_not_output():
    xyz = rays(np.linspace(-180, 180, 25), np.full(25, -85.), np.full(25, 3.))
    result = query_actions(np.vstack((xyz, [0, 0, -2])), [1]*26, 2)
    assert result['point_count'].tolist() == [0, 0, 0, 25]
    np.testing.assert_array_equal(result['direction_frd'][3], [0, 0, 1])
    assert result['window_deg'] == 20 and result['cone_half_angle_deg'] == 10
    assert action_membership([[0, 0, -2]]).tolist() == [[False, False, False, False]]


def test_kde_counts_actual_points_and_ties():
    result = query_actions([[1, 0, 0]]+[[7, 0, 0]]*10, [0]+[1]*10, 2)
    assert result['depth_m'][0] > 6.9 and result['depth_min_m'][0] == 1
    np.testing.assert_array_equal(result['semantic_votes'][0], [1, 10])
    assert result['semantic_mode_fraction'][0] == pytest.approx(10/11)
    assert kde_depth_peak([2, 2, 10, 10]) == pytest.approx(2, abs=1e-5)
    tied = query_actions([[3, 0, 0], [4, 0, 0]], [1, 0], 2)
    assert tied['category_id'][0] == 0


def test_20_degree_window_is_wider_than_old_10_and_radius_is_15():
    point = rays([7.5], [0], [4])
    assert query_actions(point, [1], 2)['observed'][0]
    assert not query_actions(point, [1], 2, window_deg=10)['observed'][0]
    result = query_actions([[15, 0, 0], [15.001, 0, 0]], [1, 1], 2)
    assert result['point_count'][0] == 1 and result['radius_m'] == 15
    assert not result['full_sphere_allocated']


def test_empty_points():
    empty = query_actions(np.empty((0, 3)), np.empty(0, int), 2)
    assert np.isnan(empty['depth_m']).all() and (empty['category_id'] == -1).all()
    assert empty['depth_m'].shape == (4,)


@pytest.mark.parametrize('directions', [[], {}, [dict(name='x', yaw_deg=float('nan'), elevation_deg=0)],
    [dict(name='x', yaw_deg=181, elevation_deg=0)], [dict(name='x', yaw_deg=0, elevation_deg=-91)],
    [dict(name='', yaw_deg=0, elevation_deg=0)], [dict(name='x', yaw_deg=0)],
    [dict(name='x', yaw_deg=0, elevation_deg=0)]*2])
def test_invalid_direction_configuration_is_rejected(directions):
    from src.actions import validate_directions
    with pytest.raises(ValueError):
        validate_directions(directions)
