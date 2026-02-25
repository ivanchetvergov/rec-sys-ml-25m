'use client'

/**
 * userStore — client-side localStorage store for:
 *   - watched:   movies the user marked as watched
 *   - watchlist: movies the user wants to watch
 *   - reviews:   ratings + text reviews (same format as MovieDetailModal uses)
 *
 * All functions are synchronous and safe to call during SSR (they return
 * empty/null when window is undefined).
 */

export interface UserReview {
	movieId: number
	rating: number // 1–5 stars
	review: string
	savedAt: string // ISO date
}

export interface WatchedEntry {
	movieId: number
	watchedAt: string
}

export interface WatchlistEntry {
	movieId: number
	addedAt: string
}

// ── Keys ────────────────────────────────────────────────────────────────────
const WATCHED_KEY = 'recsys_watched'
const WATCHLIST_KEY = 'recsys_watchlist'
// reviews are stored per-movie by MovieDetailModal as `movie_review_${id}`

// ── Helpers ─────────────────────────────────────────────────────────────────
function read<T>(key: string, fallback: T): T {
	if (typeof window === 'undefined') return fallback
	try {
		const raw = localStorage.getItem(key)
		return raw ? (JSON.parse(raw) as T) : fallback
	} catch {
		return fallback
	}
}

function write<T>(key: string, value: T): void {
	if (typeof window === 'undefined') return
	try {
		localStorage.setItem(key, JSON.stringify(value))
	} catch {
		/* noop */
	}
}

// ── Watched ─────────────────────────────────────────────────────────────────
export function getWatched(): WatchedEntry[] {
	return read<WatchedEntry[]>(WATCHED_KEY, [])
}

export function addWatched(movieId: number): void {
	const list = getWatched()
	if (list.some(e => e.movieId === movieId)) return
	write(WATCHED_KEY, [
		{ movieId, watchedAt: new Date().toISOString() },
		...list,
	])
}

export function removeWatched(movieId: number): void {
	write(
		WATCHED_KEY,
		getWatched().filter(e => e.movieId !== movieId),
	)
}

export function isWatched(movieId: number): boolean {
	return getWatched().some(e => e.movieId === movieId)
}

// ── Watchlist ────────────────────────────────────────────────────────────────
export function getWatchlist(): WatchlistEntry[] {
	return read<WatchlistEntry[]>(WATCHLIST_KEY, [])
}

export function addToWatchlist(movieId: number): void {
	const list = getWatchlist()
	if (list.some(e => e.movieId === movieId)) return
	write(WATCHLIST_KEY, [
		{ movieId, addedAt: new Date().toISOString() },
		...list,
	])
}

export function removeFromWatchlist(movieId: number): void {
	write(
		WATCHLIST_KEY,
		getWatchlist().filter(e => e.movieId !== movieId),
	)
}

export function isInWatchlist(movieId: number): boolean {
	return getWatchlist().some(e => e.movieId === movieId)
}

// ── Reviews (read-only aggregation from MovieDetailModal's keys) ─────────────
export function getAllReviews(): UserReview[] {
	if (typeof window === 'undefined') return []
	const reviews: UserReview[] = []
	for (let i = 0; i < localStorage.length; i++) {
		const key = localStorage.key(i)
		if (!key?.startsWith('movie_review_')) continue
		try {
			const movieId = parseInt(key.replace('movie_review_', ''), 10)
			const raw = localStorage.getItem(key)
			if (!raw) continue
			const parsed = JSON.parse(raw) as {
				rating?: number
				review?: string
				savedAt?: string
			}
			if ((parsed.rating ?? 0) > 0 || (parsed.review ?? '').trim()) {
				reviews.push({
					movieId,
					rating: parsed.rating ?? 0,
					review: parsed.review ?? '',
					savedAt: parsed.savedAt ?? '',
				})
			}
		} catch {
			/* skip corrupt entries */
		}
	}
	return reviews.sort((a, b) => (b.savedAt > a.savedAt ? 1 : -1))
}

export function getReview(movieId: number): UserReview | null {
	if (typeof window === 'undefined') return null
	try {
		const raw = localStorage.getItem(`movie_review_${movieId}`)
		if (!raw) return null
		const p = JSON.parse(raw) as {
			rating?: number
			review?: string
			savedAt?: string
		}
		if ((p.rating ?? 0) === 0 && !(p.review ?? '').trim()) return null
		return {
			movieId,
			rating: p.rating ?? 0,
			review: p.review ?? '',
			savedAt: p.savedAt ?? '',
		}
	} catch {
		return null
	}
}

// ── Genre stats (from watched + reviews) ─────────────────────────────────────
export interface GenreStat {
	genre: string
	count: number
}

export function getGenreStats(
	movieGenres: Record<number, string | null>,
): GenreStat[] {
	const counts: Record<string, number> = {}
	Object.values(movieGenres).forEach(genres => {
		if (!genres) return
		genres.split('|').forEach(g => {
			counts[g] = (counts[g] ?? 0) + 1
		})
	})
	return Object.entries(counts)
		.map(([genre, count]) => ({ genre, count }))
		.sort((a, b) => b.count - a.count)
		.slice(0, 8)
}
