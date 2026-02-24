from typing import Optional
from pydantic import BaseModel


class Movie(BaseModel):
    id: int
    title: str
    genres: Optional[str]
    year: Optional[int]
    avg_rating: Optional[float]
    num_ratings: Optional[int]
    popularity_score: Optional[float]
    tmdb_id: Optional[int] = None
    imdb_id: Optional[str] = None


class PopularMoviesResponse(BaseModel):
    total_returned: int
    offset: int
    movies: list[Movie]


class MovieDetails(BaseModel):
    """TMDB-enriched details for a single movie."""
    id: int
    title: str
    overview: Optional[str] = None
    tagline: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    runtime: Optional[int] = None
    tmdb_rating: Optional[float] = None
    tmdb_votes: Optional[int] = None
    release_date: Optional[str] = None
