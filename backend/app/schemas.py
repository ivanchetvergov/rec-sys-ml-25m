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


class PopularMoviesResponse(BaseModel):
    total_returned: int
    offset: int
    movies: list[Movie]
