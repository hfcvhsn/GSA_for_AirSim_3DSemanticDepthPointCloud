"""CPU-only four-direction point votes and radial KDE; no rendering or ML.

Extracted from OJUAV/tests/spherical_gsa.py and its small query wrapper.
Kept here so the RGB-D -> action pipeline has no external OJUAV dependency.
Body FRD meters, fixed 15 m radius, 20-degree cone windows by default.
"""
import math
import numpy as np

RADIUS_M = 15.0
DEFAULT_WINDOW_DEG = 20.0  # Full angular diameter, 10-degree half-angle.
ACTION_NAMES = ("forward", "yaw_left_30", "yaw_right_30", "down")
ACTION_LABELS = ("向前", "向左偏30度", "向右偏30度", "竖直向下")
YAW_DEG = np.array([0., -30., 30., 0.])
ELEVATION_DEG = np.array([0., 0., 0., -90.])
DIRECTION_FRD = np.array([[1., 0., 0.], [np.sqrt(3)/2, -.5, 0.],
                          [np.sqrt(3)/2, .5, 0.], [0., 0., 1.]])
ACTION_DISPLAY_ORDER = (('front', 0), ('L30', 1), ('R30', 2), ('DOWN', 3))
DEFAULT_DIRECTIONS = tuple(dict(name=name, yaw_deg=float(yaw), elevation_deg=float(elevation))
                           for (name, _), yaw, elevation in zip(ACTION_DISPLAY_ORDER, YAW_DEG, ELEVATION_DEG))


def validate_directions(directions=None):
    """Normalize named yaw/elevation queries; yaw is positive right, elevation up."""
    if directions is None:
        directions = DEFAULT_DIRECTIONS
    if not isinstance(directions, (list, tuple)) or not directions:
        raise ValueError('directions must be a nonempty list of name/yaw_deg/elevation_deg objects')
    result = []
    for item in directions:
        if not isinstance(item, dict) or set(item) != {'name', 'yaw_deg', 'elevation_deg'}:
            raise ValueError('Each direction requires exactly name, yaw_deg and elevation_deg')
        name = item['name']
        try:
            yaw, elevation = float(item['yaw_deg']), float(item['elevation_deg'])
        except (TypeError, ValueError):
            raise ValueError('Direction angles must be finite numbers') from None
        if (not isinstance(name, str) or not name.strip() or name != name.strip() or
                not math.isfinite(yaw) or not -180 <= yaw <= 180 or
                not math.isfinite(elevation) or not -90 <= elevation <= 90):
            raise ValueError('Require a nonempty name, yaw in [-180, 180] and elevation in [-90, 90]')
        result.append(dict(name=name, yaw_deg=yaw, elevation_deg=elevation))
    if len({item['name'] for item in result}) != len(result):
        raise ValueError('Direction names must be unique')
    return tuple(result)


def direction_vectors(directions):
    directions = validate_directions(directions)
    yaw = np.radians([item['yaw_deg'] for item in directions])
    elevation = np.radians([item['elevation_deg'] for item in directions])
    return np.column_stack((np.cos(elevation)*np.cos(yaw), np.cos(elevation)*np.sin(yaw),
                            -np.sin(elevation)))


def action_membership(xyz, window_deg=DEFAULT_WINDOW_DEG):
    """N x 4 cap membership; poles use angular distance, without a yaw seam."""
    if not np.isfinite(window_deg) or not 0 < window_deg <= 180:
        raise ValueError('Require finite 0 < window_deg <= 180')
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('Expected Nx3 points in the action FRD frame')
    distance = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (distance > 1e-8) & (distance <= RADIUS_M)
    threshold = distance[:, None] * np.cos(np.radians(window_deg/2))
    return valid[:, None] & (xyz @ DIRECTION_FRD.T >= threshold - 1e-12)



def kde_depth_peak(distances, bandwidth=0.4):
    """Numerical global mode of an equal-point-weight 1D Gaussian KDE.

    bandwidth is Gaussian standard deviation in meters. Search all grid maxima
    (spacing <= bandwidth/4), refine each, and choose the near mode on a tie.
    Duplicate distances retain their multiplicity. Constant/singleton cells
    return their measured distance without a singular covariance calculation.
    """
    from scipy.optimize import minimize_scalar
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1 or not len(distances) or not np.isfinite(distances).all() or np.any(distances <= 0):
        raise ValueError("KDE requires a nonempty vector of finite positive distances")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("KDE bandwidth must be finite and positive")
    values, counts = np.unique(distances, return_counts=True)
    if len(values) == 1:
        return float(values[0])
    weights = counts/counts.sum()
    intervals = (values[-1]-values[0])/(bandwidth/4)
    if not np.isfinite(intervals) or intervals > 1_000_000:
        raise ValueError("KDE bandwidth too small for depth span; increase bandwidth")
    grid = np.linspace(values[0], values[-1], max(65, int(np.ceil(intervals))+1))

    def density(query):
        query = np.atleast_1d(query)
        result = np.zeros(len(query))
        # Bounded temporary arrays, including cells with many distinct depths.
        for start in range(0, len(values), 2048):
            support = values[start:start+2048]
            chunk = max(1, 262144//len(support))
            for offset in range(0, len(query), chunk):
                scaled = (query[offset:offset+chunk, None]-support[None, :])/bandwidth
                result[offset:offset+chunk] += np.exp(-.5*scaled*scaled) @ weights[start:start+2048]
        return result

    scores = density(grid)
    left, right = np.r_[-np.inf, scores[:-1]], np.r_[scores[1:], -np.inf]
    maxima = np.flatnonzero((scores >= left) & (scores >= right) & ((scores > left) | (scores > right)))
    peak_positions, peak_scores = [], []
    for index in maxima:
        lo, hi = grid[max(0, index-1)], grid[min(len(grid)-1, index+1)]
        result = minimize_scalar(lambda value: -density([value])[0], bounds=(lo, hi), method='bounded',
                                 options={'xatol': max(1e-10, bandwidth*1e-6)})
        # Include grid endpoints; refinement must not discard a boundary mode.
        peak_positions.extend((grid[index], result.x))
        peak_scores.extend((scores[index], -result.fun))
    peak_positions, peak_scores = np.asarray(peak_positions), np.asarray(peak_scores)
    tied = np.isclose(peak_scores, peak_scores.max(), rtol=1e-10, atol=1e-12)
    return float(peak_positions[tied].min())


def pool_action_samples(distance_m, categories, membership, num_classes, kde_bandwidth=.4):
    """Pool scalar ranges and pixel/action masks directly; no XYZ or voxel grid."""
    distance_m, categories, membership = np.asarray(distance_m, float), np.asarray(categories), np.asarray(membership)
    if (distance_m.ndim != 1 or categories.shape != distance_m.shape or membership.ndim != 2 or
            membership.shape[0] != len(distance_m) or membership.shape[1] < 1 or membership.dtype != bool):
        raise ValueError('Expected N ranges, N semantic IDs and boolean NxK direction membership')
    if (not np.isfinite(distance_m).all() or np.any(distance_m <= 0) or np.any(distance_m > RADIUS_M) or
            not np.isfinite(kde_bandwidth) or kde_bandwidth <= 0):
        raise ValueError('Require finite 0 < distance <= 15 and positive KDE bandwidth')
    if (not isinstance(num_classes, (int, np.integer)) or num_classes < 1 or
            not np.isfinite(categories).all() or np.any(categories < 0) or
            np.any(categories >= num_classes) or np.any(categories != np.rint(categories))):
        raise ValueError('Semantic IDs must index the GSA class table')
    labels = categories.astype(np.int32)
    count = membership.shape[1]
    depth, depth_min = np.full(count, np.nan, np.float32), np.full(count, np.nan, np.float32)
    sample_count, votes = np.zeros(count, np.int64), np.zeros((count, num_classes), np.int64)
    for action in range(count):
        inside = membership[:, action]
        d = distance_m[inside]
        sample_count[action] = len(d)
        if len(d):
            depth[action], depth_min[action] = kde_depth_peak(d, kde_bandwidth), d.min()
            votes[action] = np.bincount(labels[inside], minlength=num_classes)
    observed = sample_count > 0
    winner = votes.argmax(axis=1)
    category = np.where(observed, winner, -1).astype(np.int32)
    denominator = np.maximum(sample_count, 1)
    return dict(depth_m=depth, depth_min_m=depth_min, category_id=category, sample_count=sample_count,
                observed=observed, semantic_valid=observed & (category > 0), semantic_no_return=~observed,
                semantic_vote_ids=np.arange(num_classes, dtype=np.int32), semantic_votes=votes,
                semantic_mode_count=votes[np.arange(count), winner],
                semantic_mode_fraction=votes[np.arange(count), winner]/denominator,
                semantic_unlabeled_fraction=votes[:, 0]/denominator,
                semantic_known_fraction=votes[:, 1:].sum(axis=1)/denominator)


def format_action_rows(features, classes, directions=None):
    """Return JSON-safe rows in configured order; default front/L30/R30/DOWN."""
    rows = []
    for index, direction in enumerate(validate_directions(directions)):
        action = direction['name']
        depth = float(features['depth_m'][index])
        category = int(features['category_id'][index])
        rows.append(dict(action=action, depth_m=depth if math.isfinite(depth) else None,
                         semantic_id=category, semantic_name=classes[category] if category >= 0 else 'no_return'))
    return rows


def query_actions(xyz, categories, num_classes, window_deg=DEFAULT_WINDOW_DEG, kde_bandwidth=.4):
    """Legacy cloud-query API. The RGB-D wrapper uses scalar pixel samples instead."""
    xyz, categories = np.asarray(xyz, float), np.asarray(categories)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or categories.shape != (len(xyz),):
        raise ValueError('Expected Nx3 body-FRD points and N semantic IDs')
    # Validate even labels attached to points that will be filtered out.
    if (not isinstance(num_classes, (int, np.integer)) or num_classes < 1 or
            not np.isfinite(categories).all() or np.any(categories < 0) or
            np.any(categories >= num_classes) or np.any(categories != np.rint(categories))):
        raise ValueError('Semantic IDs must index the GSA class table')
    distance = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (distance > 1e-8) & (distance <= RADIUS_M)
    membership = action_membership(xyz, window_deg)
    features = pool_action_samples(distance[valid], categories[valid], membership[valid], num_classes, kde_bandwidth)
    features['point_count'] = features.pop('sample_count')
    features.update(action_names=np.asarray(ACTION_NAMES), action_labels=np.asarray(ACTION_LABELS),
        yaw_deg=YAW_DEG.copy(), elevation_deg=ELEVATION_DEG.copy(), direction_frd=DIRECTION_FRD.copy(),
        radius_m=np.array(RADIUS_M), window_deg=np.array(window_deg), kde_bandwidth_m=np.array(kde_bandwidth),
        input_points=np.array(len(xyz)), in_radius_points=np.array(valid.sum()), full_sphere_allocated=np.array(False),
        pooling=np.array('kde-mode'), semantic_pooling=np.array('point-majority'),
        feature_schema=np.array('four-action-cones-v5'), window_shape=np.array('spherical_cap'),
        cone_half_angle_deg=np.array(window_deg/2), action_membership=membership)
    return features


def get_action_depth_semantics(xyz, semantic_id, classes, *, window_deg=DEFAULT_WINDOW_DEG, kde_bandwidth=.4):
    """Legacy cloud-query interface; null/-1/no_return marks an empty cone."""
    return format_action_rows(query_actions(xyz, semantic_id, len(classes), window_deg, kde_bandwidth), classes)
