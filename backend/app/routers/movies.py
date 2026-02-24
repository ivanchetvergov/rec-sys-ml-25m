from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas import Movie, MovieDetails, PersonalRecsResponse, PopularMoviesResponse
from app.services.popularity_service import PopularityService, get_popularity_service
from app.services.recommender_service import RecommenderService, get_recommender_service
from app.services.tmdb_service import TMDBService, get_tmdb_service

router = APIRouter(prefix="/movies", tags=["movies"])


@router.get("/popular", response_model=PopularMoviesResponse)
def popular_movies(
    limit: int = Query(20, ge=1, le=20_000, description="Number of movies to return"),
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
        total_available=service.total_count(),
        movies=movies,
    )


@router.get("/personal", response_model=PersonalRecsResponse)
def personal_recommendations(
    user_id: int = Query(..., description="MovieLens user id"),
    limit: int = Query(20, ge=1, le=100, description="Number of recommendations"),
    rec_service: RecommenderService = Depends(get_recommender_service),
    pop_service: PopularityService = Depends(get_popularity_service),
):
    """
    Returns personalised movie recommendations for the given user.

    - If the user is in the training set, uses the two-stage iALS + CatBoost ranker.
    - Cold-start (unknown user) or model not trained yet: falls back to popularity ranking.
    """
    movies, model_name = rec_service.get_personal_recs(
        user_id=user_id,
        n=limit,
        pop_fallback=pop_service,
    )
    return PersonalRecsResponse(
        user_id=user_id,
        model=model_name,
        total_returned=len(movies),
        movies=movies,
    )


@router.get("/{movie_id}/details", response_model=MovieDetails)
async def movie_details(
    movie_id: int,
    service: PopularityService = Depends(get_popularity_service),
    tmdb: TMDBService = Depends(get_tmdb_service),
):
    """
    Returns TMDB-enriched details for a single movie: poster, overview,
    tagline, runtime, TMDB rating.

    Requires TMDB_API_KEY env var. If the key is absent or the request
    fails, the endpoint still returns 200 with null poster/overview fields.
    """
    tmdb_id = service.get_tmdb_id(movie_id)
    if tmdb_id is None:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found or has no TMDB id")

    extra = await tmdb.get_movie_details(tmdb_id) or {}

    return MovieDetails(
        id=movie_id,
        title="",  # caller already knows the title from the movie list
        overview=extra.get("overview"),
        poster_url=extra.get("poster_url"),
        backdrop_url=extra.get("backdrop_url"),
        tagline=extra.get("tagline"),
        runtime=extra.get("runtime"),
        tmdb_rating=extra.get("tmdb_rating"),
        tmdb_votes=extra.get("tmdb_votes"),
        release_date=extra.get("release_date"),
    )


@router.get("/{movie_id}", response_model=Movie)
def get_movie(
    movie_id: int,
    service: PopularityService = Depends(get_popularity_service),
):
    """Return a single movie by its MovieLens movie_id."""
    movie = service.get_movie(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    return movie
