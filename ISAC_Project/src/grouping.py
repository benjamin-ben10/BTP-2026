import numpy as np
from sklearn.cluster import KMeans

def random_grouping(M, Q, seed=42):

    rng = np.random.default_rng(seed)

    labels = rng.integers(
        0,
        Q,
        size=M
    )

    groups = []

    for q in range(Q):
        groups.append(
            np.where(labels == q)[0]
        )

    return groups

def adjacent_grouping(M, Q):

    indices = np.arange(M)

    groups = np.array_split(
        indices,
        Q
    )

    return groups

def phase_grouping(
        cascaded_channel,
        Q):

    phases = np.angle(
        cascaded_channel
    )

    bins = np.linspace(
        -np.pi,
        np.pi,
        Q + 1
    )

    labels = np.digitize(
        phases,
        bins
    ) - 1

    groups = []

    for q in range(Q):
        groups.append(
            np.where(labels == q)[0]
        )

    return groups

def sbl_grouping(cascaded_channel,
                 Q=250,
                 max_iter=50):

    M = len(cascaded_channel)

    # Feature vector

    X = np.column_stack([
        np.abs(cascaded_channel),
        np.angle(cascaded_channel)
    ])

    # Initial variance
    gamma = np.ones(M)

    for _ in range(max_iter):

        Gamma = np.diag(gamma)

        mu = Gamma @ np.linalg.pinv(Gamma + np.eye(M))

        gamma_new = np.real(np.diag(mu))

        if np.linalg.norm(gamma_new-gamma) < 1e-5:
            break

        gamma = gamma_new

    # dominant elements

    importance = gamma

    # cluster according to importance

    features = np.column_stack([
        importance,
        np.angle(cascaded_channel)
    ])

    kmeans = KMeans(
        n_clusters=Q,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(features)

    groups = []

    for q in range(Q):

        groups.append(
            np.where(labels == q)[0]
        )

    return groups

def hybrid_grouping(
        cascaded_channel,
        Q=250,
        mag_weight=0.7,
        phase_weight=0.3):

    magnitude = np.abs(
        cascaded_channel
    )

    phase = np.angle(
        cascaded_channel
    )

    magnitude = (
        magnitude - magnitude.min()
    ) / (
        magnitude.max() - magnitude.min()
    )

    phase = (
        phase + np.pi
    ) / (
        2*np.pi
    )

    features = np.column_stack([
        mag_weight*magnitude,
        phase_weight*phase
    ])

    kmeans = KMeans(
        n_clusters=Q,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        features
    )

    groups = []

    for q in range(Q):

        groups.append(
            np.where(labels == q)[0]
        )

    return groups

def grouped_channel(
        h_BI,
        h_IU,
        groups):

    h_eff = 0

    for idx in groups:

        cascaded = h_BI[idx] * h_IU[idx]

        phase = np.exp(
            -1j*np.angle(
                np.sum(cascaded)
            )
        )

        h_eff += phase*np.sum(cascaded)

    return h_eff