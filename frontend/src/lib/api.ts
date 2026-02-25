/**
 * SSR (Server Components / Route Handlers): use the internal Docker hostname
 *   BACKEND_INTERNAL_URL=http://backend:8000
 * Browser (Client Components): use the public origin served by nginx
 *   NEXT_PUBLIC_API_URL=http://localhost  (or the real domain in prod)
 *
 * NEXT_PUBLIC_* vars are inlined at build time → available in the browser.
 * Non-public vars (BACKEND_INTERNAL_URL) are only visible on the server.
 */
const API_URL =
	typeof window === 'undefined'
		? (process.env.BACKEND_INTERNAL_URL ?? 'http://localhost:8000') // server-side
		: (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost') // client-side

export interface Movie {
	id: number
	title: string
	genres: string | null
	year: number | null
	avg_rating: number | null
	num_ratings: number | null
	popularity_score: number | null
	tmdb_id: number | null
	imdb_id: string | null
}

export interface PopularMoviesResponse {
	total_returned: number
	offset: number
	total_available: number | null
	movies: Movie[]
}

/** Enriched details fetched from TMDB via the backend proxy. */
export interface MovieDetails {
	id: number
	title: string
	overview: string | null
	poster_url: string | null
	backdrop_url: string | null
	tagline: string | null
	runtime: number | null
	tmdb_rating: number | null
	tmdb_votes: number | null
	release_date: string | null
}

export async function fetchPopularMovies(
	limit = 20,
	offset = 0,
): Promise<PopularMoviesResponse> {
	const res = await fetch(
		`${API_URL}/api/movies/popular?limit=${limit}&offset=${offset}`,
		{
			next: { revalidate: 3600 }, // кешируем на 1 час (Next.js ISR)
		},
	)
	if (!res.ok) throw new Error('Failed to fetch popular movies')
	return res.json()
}

/**
 * Fetch the full movie catalog from the backend (no ISR cache — client-side use only).
 * Returns all movies sorted by popularity.
 */
export async function fetchAllMovies(): Promise<Movie[]> {
	const res = await fetch(
		`${API_URL}/api/movies/popular?limit=20000&offset=0`,
		{
			cache: 'no-store',
		},
	)
	if (!res.ok) throw new Error('Failed to fetch catalog')
	const data: PopularMoviesResponse = await res.json()
	return data.movies
}

/** Fetch TMDB-enriched details for a single movie. Never throws — returns null on error. */
export async function fetchMovieDetails(
	movieId: number,
): Promise<MovieDetails | null> {
	try {
		const res = await fetch(`${API_URL}/api/movies/${movieId}/details`, {
			next: { revalidate: 86400 }, // cache for 24h
		})
		if (!res.ok) return null
		return res.json()
	} catch {
		return null
	}
}

/** Fetch a single movie by id (genres, year, ratings etc.). Never throws. */
export async function fetchMovie(movieId: number): Promise<Movie | null> {
	try {
		const res = await fetch(`${API_URL}/api/movies/${movieId}`, {
			next: { revalidate: 86400 },
		})
		if (!res.ok) return null
		return res.json()
	} catch {
		return null
	}
}

/** A personally recommended movie with a relevance score. */
export interface PersonalRec {
	id: number
	score: number
	title: string | null
	genres: string | null
	year: number | null
	avg_rating: number | null
	num_ratings: number | null
	popularity_score: number | null
	tmdb_id: number | null
}

export interface PersonalRecsResponse {
	user_id: number
	/** "two_stage" when the ML model is used, "popularity_fallback" otherwise. */
	model: string
	total_returned: number
	movies: PersonalRec[]
}

/**
 * Fetch personal recommendations for a given user.
 * Falls back gracefully — never throws.
 */
export async function fetchPersonalRecs(
	userId: number,
	limit = 20,
): Promise<PersonalRecsResponse | null> {
	try {
		const res = await fetch(
			`${API_URL}/api/movies/personal?user_id=${userId}&limit=${limit}`,
			{ cache: 'no-store' }, // personalised — never cache
		)
		if (!res.ok) return null
		return res.json()
	} catch {
		return null
	}
}

export interface SimilarMoviesResult {
	movies: Movie[]
	/** "als_cosine" | "genre_jaccard" | "not_available" */
	model: string
}

/**
 * Fetch item-item similar movies for a given movieId.
 * Uses ALS cosine similarity index on the backend (falls back to genre Jaccard).
 * Never throws — returns empty array on error.
 */
export async function fetchSimilarMovies(
	movieId: number,
	limit = 20,
): Promise<SimilarMoviesResult> {
	try {
		const res = await fetch(
			`${API_URL}/api/movies/${movieId}/similar?limit=${limit}`,
			{ cache: 'no-store' },
		)
		if (!res.ok) return { movies: [], model: '' }
		const data = await res.json()
		return { movies: data.movies ?? [], model: data.model ?? '' }
	} catch {
		return { movies: [], model: '' }
	}
}
