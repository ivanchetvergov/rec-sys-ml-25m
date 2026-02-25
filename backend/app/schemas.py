from typing import List, Optional
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
    total_available: Optional[int] = None
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


class PersonalRec(BaseModel):
    """A single personally recommended movie."""
    id: int
    score: float
    title: Optional[str] = None
    genres: Optional[str] = None
    year: Optional[int] = None
    avg_rating: Optional[float] = None
    num_ratings: Optional[int] = None
    popularity_score: Optional[float] = None
    tmdb_id: Optional[int] = None


class PersonalRecsResponse(BaseModel):
    """Response for personal recommendation endpoint."""
    user_id: int
    model: str          # "two_stage" | "popularity_fallback"
    total_returned: int
    movies: List[PersonalRec]


class SimilarMoviesResponse(BaseModel):
    movie_id: int
    model: str          # "als_cosine" | "genre_jaccard" | "not_available"
    total_returned: int
    movies: List[Movie]
