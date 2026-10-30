"""Gets the artifact store for a REST route that needs it."""

from fastapi import HTTPException

from midrun.runtime.artifacts import ArtifactStore


def require_artifact_store(store: ArtifactStore | None) -> ArtifactStore:
    """Returns `store`, the run's artifact store.

    Raises:
        HTTPException: The store did not start, for example because its directory
            could not be created (status 503).
    """
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="midrun: the artifact store is not available; it did not start"
            " (see midrun.log)",
        )
    return store
