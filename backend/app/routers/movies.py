from fastapi import APIRouter, Depends, Query

from app.schemas import PopularMoviesResponse
from app.services.popularity_service import PopularityService, get_popularity_service

router = APIRouter(prefix="/movies", tags=["movies"])


@router.get("/popular", response_model=PopularMoviesResponse)
def popular_movies(
    limit: int = Query(20, ge=1, le=100, description="Number of movies to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    service: PopularityService = Depends(get_popularity_service),
):
    """
    Returns the most popular movies sorted by popularity score.

    popularity = avg_rating × log(1 + num_ratings)
    """
    movies = service.get_popular(limit=limit, offset=offset)
    return PopularMoviesResponse(
        total_returned=len(movies),
        offset=offset,
        movies=movies,
    )
