"""FastAPI entrypoint for PathSnap distributed map-matching & route reconstruction.

Run locally or across two laptops via:
    uvicorn backend.app.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.config import load_api_settings
from backend.app.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    TripListResponse,
    TripMapDataResponse,
)
from backend.app.api.service import (
    InferenceService,
    InvalidTripIdError,
    TripNotFoundError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pathsnap.api")


def create_app() -> FastAPI:
    """Application factory supporting clean testing and production startup."""
    settings = load_api_settings()
    service = InferenceService(settings)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        service.startup()
        app_instance.state.inference_service = service
        yield

    app = FastAPI(
        title="PathSnap Map-Matching & Route Reconstruction API",
        description=(
            "Distributed ML inference backend running spatial candidate retrieval, "
            "26-feature engineering, tuned HistGradientBoosting scoring, and HMM/Viterbi "
            "route reconstruction. Strictly isolated from reference .route files."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t_req = time.perf_counter()
        client_host = request.client.host if request.client else "unknown"
        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - t_req) * 1000.0
            logger.info(
                "HTTP %s %s | client=%s | status=%d | %.1fms",
                request.method,
                request.url.path,
                client_host,
                response.status_code,
                elapsed_ms,
            )
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t_req) * 1000.0
            logger.exception(
                "HTTP %s %s | client=%s | UNHANDLED ERROR after %.1fms: %s",
                request.method,
                request.url.path,
                client_host,
                elapsed_ms,
                exc,
            )
            raise

    def _get_service(request: Request) -> InferenceService:
        svc = getattr(request.app.state, "inference_service", None)
        if svc is None:
            raise HTTPException(status_code=503, detail="Inference service not initialized.")
        return svc

    @app.exception_handler(InvalidTripIdError)
    async def invalid_trip_handler(request: Request, exc: InvalidTripIdError) -> JSONResponse:
        logger.warning("Rejected invalid trip request %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(detail=str(exc), error_code="INVALID_TRIP_ID").model_dump(),
        )

    @app.exception_handler(TripNotFoundError)
    async def not_found_handler(request: Request, exc: TripNotFoundError) -> JSONResponse:
        logger.warning("Trip not found for %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(detail=str(exc), error_code="TRIP_NOT_FOUND").model_dump(),
        )

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health_check(request: Request) -> HealthResponse:
        svc = _get_service(request)
        return svc.get_health()

    @app.get("/trips", response_model=TripListResponse, tags=["Trips"])
    def list_trips(request: Request) -> TripListResponse:
        svc = _get_service(request)
        return svc.list_trips()

    @app.get(
        "/trips/{trip_id}/map-data",
        response_model=TripMapDataResponse,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
        tags=["Trips"],
    )
    def trip_map_data(trip_id: str, request: Request) -> TripMapDataResponse:
        svc = _get_service(request)
        return svc.get_trip_map_data(trip_id)

    @app.post(
        "/predict/{trip_id}",
        response_model=PredictionResponse,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
        tags=["Inference"],
    )
    def predict_route(trip_id: str, request: Request) -> PredictionResponse:
        svc = _get_service(request)
        return svc.predict_trip(trip_id)

    return app


app = create_app()
