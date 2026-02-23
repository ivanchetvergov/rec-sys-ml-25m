const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Movie {
    id: number;
    title: string;
    genres: string | null;
    year: number | null;
    avg_rating: number | null;
    num_ratings: number | null;
    popularity_score: number | null;
}

export interface PopularMoviesResponse {
    total_returned: number;
    offset: number;
    movies: Movie[];
}

export async function fetchPopularMovies(limit = 20, offset = 0): Promise<PopularMoviesResponse> {
    const res = await fetch(`${API_URL}/api/movies/popular?limit=${limit}&offset=${offset}`, {
        next: { revalidate: 3600 }, // кешируем на 1 час (Next.js ISR)
    });
    if (!res.ok) throw new Error("Failed to fetch popular movies");
    return res.json();
}
