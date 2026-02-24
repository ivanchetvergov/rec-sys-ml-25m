const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Movie {
    id: number;
    title: string;
    genres: string | null;
    year: number | null;
    avg_rating: number | null;
    num_ratings: number | null;
    popularity_score: number | null;
    tmdb_id: number | null;
    imdb_id: string | null;
}

export interface PopularMoviesResponse {
    total_returned: number;
    offset: number;
    movies: Movie[];
}

/** Enriched details fetched from TMDB via the backend proxy. */
export interface MovieDetails {
    id: number;
    title: string;
    overview: string | null;
    poster_url: string | null;
    tagline: string | null;
    runtime: number | null;
    tmdb_rating: number | null;
    tmdb_votes: number | null;
    release_date: string | null;
}

export async function fetchPopularMovies(limit = 20, offset = 0): Promise<PopularMoviesResponse> {
    const res = await fetch(`${API_URL}/api/movies/popular?limit=${limit}&offset=${offset}`, {
        next: { revalidate: 3600 }, // кешируем на 1 час (Next.js ISR)
    });
    if (!res.ok) throw new Error("Failed to fetch popular movies");
    return res.json();
}

/** Fetch TMDB-enriched details for a single movie. Never throws — returns null on error. */
export async function fetchMovieDetails(movieId: number): Promise<MovieDetails | null> {
    try {
        const res = await fetch(`${API_URL}/api/movies/${movieId}/details`, {
            next: { revalidate: 86400 }, // cache for 24h
        });
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    }
}

/** Fetch a single movie by id (genres, year, ratings etc.). Never throws. */
export async function fetchMovie(movieId: number): Promise<Movie | null> {
    try {
        const res = await fetch(`${API_URL}/api/movies/${movieId}`, {
            next: { revalidate: 86400 },
        });
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    }
}
