from fastapi import APIRouter

from pyrai.api.routes.auth import router as auth_router
from pyrai.api.routes.models import router as models_router
from pyrai.api.routes.tasks import router as tasks_router
from pyrai.api.routes.users import router as users_router
from pyrai.api.routes.monitoring import router as monitoring_router

# Create main router
router = APIRouter()

# Include sub-routers
router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
router.include_router(models_router, prefix="/models", tags=["Models"])
router.include_router(tasks_router, prefix="/tasks", tags=["Tasks"])
router.include_router(users_router, prefix="/users", tags=["Users"])
router.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring"]) 