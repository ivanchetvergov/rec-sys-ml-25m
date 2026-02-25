'use client'

/**
 * MoviePageInteractive — client-side interactive section for the movie detail page.
 *
 * Includes:
 *  - Star rating + Watched + Watchlist buttons
 *  - Write/edit review textarea with Save
 *  - Saved reviews carousel (localStorage-based)
 *  - Similar movies row (genre-matched, fetched from catalog)
 */
import type { Movie } from '@/lib/api'
import { fetchPopularMovies } from '@/lib/api'
import {
    addToWatchlist,
    addWatched,
    isInWatchlist,
    isWatched,
    removeFromWatchlist,
    removeWatched,
} from '@/lib/userStore'
import Link from 'next/link'
import { useCallback, useEffect, useRef, useState } from 'react'

// ── Types ────────────────────────────────────────────────────────────────────
interface SavedReview {
    rating: number
    review: string
    savedAt: string
}

interface SimMovie {
    id: number
    title: string
    year: number | null
    avg_rating: number | null
    genres: string | null
    tmdb_id: number | null
}

// ── StarRating ───────────────────────────────────────────────────────────────
function StarRating({
    value,
    onChange,
    size = 'md',
}: {
    value: number
    onChange?: (v: number) => void
    size?: 'sm' | 'md' | 'lg'
}) {
    const [hovered, setHovered] = useState(0)
    const sizeClass = size === 'lg' ? 'text-3xl' : size === 'sm' ? 'text-lg' : 'text-2xl'
    return (
        <div className='flex gap-0.5'>
            {[1, 2, 3, 4, 5].map(star => (
                <button
                    key={star}
                    type='button'
                    onMouseEnter={() => onChange && setHovered(star)}
                    onMouseLeave={() => onChange && setHovered(0)}
                    onClick={() => onChange?.(star)}
                    disabled={!onChange}
                    className={`${sizeClass} transition-transform ${onChange ? 'hover:scale-110 cursor-pointer' : 'cursor-default'} focus:outline-none`}
                    style={{ color: star <= (hovered || value) ? '#f59e0b' : 'rgba(255,255,255,0.18)' }}
                >
                    ★
                </button>
            ))}
        </div>
    )
}

// ── SimilarMovieCard ─────────────────────────────────────────────────────────
function SimilarMovieCard({ movie }: { movie: SimMovie }) {
    const [posterUrl, setPosterUrl] = useState<string | null>(null)
    const gradient = `linear-gradient(135deg, hsl(${(movie.id * 37) % 360},40%,22%), hsl(${(movie.id * 37 + 120) % 360},35%,13%))`

    useEffect(() => {
        if (!movie.tmdb_id) return
        fetch(`/api/movies/${movie.id}/details`)
            .then(r => r.ok ? r.json() : null)
            .then(d => d?.poster_url && setPosterUrl(d.poster_url))
            .catch(() => { })
    }, [movie.id, movie.tmdb_id])

    return (
        <Link href={`/movies/${movie.id}`} className='flex-shrink-0 w-36 group cursor-pointer'>
            <div
                className='relative rounded-xl overflow-hidden shadow-lg'
                style={{ aspectRatio: '2/3', background: gradient, border: '1px solid rgba(255,255,255,0.07)' }}
            >
                {posterUrl ? (
                    <img src={posterUrl} alt={movie.title} className='w-full h-full object-cover transition-transform duration-300 group-hover:scale-105' />
                ) : (
                    <div className='absolute inset-0 flex items-end p-2'>
                        <span className='text-white text-xs font-semibold leading-snug drop-shadow'>{movie.title}</span>
                    </div>
                )}
                <div className='absolute inset-0 bg-black opacity-0 group-hover:opacity-20 transition-opacity' />
            </div>
            <p className='mt-2 text-xs text-zinc-300 font-medium truncate leading-snug'>{movie.title}</p>
            {movie.year && <p className='text-xs text-zinc-500'>{movie.year}</p>}
            {movie.avg_rating && (
                <p className='text-xs text-yellow-400 font-semibold'>★ {movie.avg_rating.toFixed(1)}</p>
            )}
        </Link>
    )
}

// ── Main component ────────────────────────────────────────────────────────────
interface Props {
    movie: Movie
}

export default function MoviePageInteractive({ movie }: Props) {
    // ── user actions
    const [watched, setWatched] = useState(false)
    const [inWatchlist, setInWatchlist] = useState(false)
    const [userRating, setUserRating] = useState(0)
    const [review, setReview] = useState('')
    const [saved, setSaved] = useState(false)

    // ── saved reviews list for this movie
    const [savedReviews, setSavedReviews] = useState<SavedReview[]>([])

    // ── similar movies
    const [similar, setSimilar] = useState<SimMovie[]>([])
    const scrollRef = useRef<HTMLDivElement>(null)

    const storageKey = `movie_review_${movie.id}`

    // ── Init from localStorage
    useEffect(() => {
        setWatched(isWatched(movie.id))
        setInWatchlist(isInWatchlist(movie.id))
        try {
            const raw = localStorage.getItem(storageKey)
            if (raw) {
                const parsed = JSON.parse(raw) as SavedReview
                setUserRating(parsed.rating ?? 0)
                setReview(parsed.review ?? '')
            }
        } catch { }
    }, [movie.id, storageKey])

    // ── Load all reviews for this movie from localStorage
    const loadReviews = useCallback(() => {
        if (typeof window === 'undefined') return
        const reviews: SavedReview[] = []
        // support both the single key and future multi-user keys
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i)
            if (!key) continue
            // match `movie_review_${movie.id}` or `movie_review_${movie.id}_user_*`
            if (key === storageKey || key.startsWith(`${storageKey}_user_`)) {
                try {
                    const val = JSON.parse(localStorage.getItem(key) ?? '')
                    if (val?.savedAt) reviews.push(val as SavedReview)
                } catch { }
            }
        }
        reviews.sort((a, b) => new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime())
        setSavedReviews(reviews)
    }, [storageKey])

    useEffect(() => { loadReviews() }, [loadReviews])

    // ── Load similar movies (genre-based)
    useEffect(() => {
        const genres = movie.genres?.split('|') ?? []
        if (genres.length === 0) return
        fetchPopularMovies(100, 0)
            .then(res => {
                const filtered = res.movies.filter(m => {
                    if (m.id === movie.id) return false
                    const mg = m.genres?.split('|') ?? []
                    return mg.some(g => genres.includes(g))
                })
                setSimilar(filtered.slice(0, 24))
            })
            .catch(() => { })
    }, [movie.id, movie.genres])

    // ── Save review
    const handleSave = () => {
        try {
            localStorage.setItem(storageKey, JSON.stringify({
                rating: userRating,
                review,
                savedAt: new Date().toISOString(),
            }))
            setSaved(true)
            loadReviews()
            setTimeout(() => setSaved(false), 2000)
        } catch { }
    }

    // ── Horizontal scroll helpers
    const scrollBy = (dir: -1 | 1) => {
        scrollRef.current?.scrollBy({ left: dir * 600, behavior: 'smooth' })
    }

    const sectionBox = 'rounded-2xl p-6'
    const sectionStyle = { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }

    return (
        <div className='flex flex-col gap-8'>
            {/* ── Your rating + Watched + Watchlist ────────────────────────── */}
            <div className={sectionBox} style={sectionStyle}>
                <h2 className='text-xs uppercase tracking-widest text-zinc-500 mb-4'>Your rating</h2>
                <div className='flex flex-wrap items-center gap-4'>
                    <StarRating value={userRating} onChange={setUserRating} size='lg' />

                    {/* Watched */}
                    <button
                        onClick={() => {
                            if (watched) { removeWatched(movie.id); setWatched(false) }
                            else { addWatched(movie.id); setWatched(true) }
                        }}
                        className='flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all'
                        style={{
                            background: watched ? 'rgba(46,160,67,0.18)' : 'rgba(255,255,255,0.06)',
                            border: `1px solid ${watched ? 'rgba(46,160,67,0.5)' : 'rgba(255,255,255,0.12)'}`,
                            color: watched ? '#4ade80' : '#a1a1aa',
                        }}
                    >
                        {watched
                            ? <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor' className='w-4 h-4'><path fillRule='evenodd' d='M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z' clipRule='evenodd' /></svg>
                            : <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor' className='w-4 h-4'><path d='M10.75 4.75a.75.75 0 0 0-1.5 0v4.5h-4.5a.75.75 0 0 0 0 1.5h4.5v4.5a.75.75 0 0 0 1.5 0v-4.5h4.5a.75.75 0 0 0 0-1.5h-4.5v-4.5Z' /></svg>
                        }
                        {watched ? 'Watched' : 'Mark watched'}
                    </button>

                    {/* Watchlist */}
                    <button
                        onClick={() => {
                            if (inWatchlist) { removeFromWatchlist(movie.id); setInWatchlist(false) }
                            else { addToWatchlist(movie.id); setInWatchlist(true) }
                        }}
                        className='flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all'
                        style={{
                            background: inWatchlist ? 'rgba(229,9,20,0.13)' : 'rgba(255,255,255,0.06)',
                            border: `1px solid ${inWatchlist ? 'rgba(229,9,20,0.4)' : 'rgba(255,255,255,0.12)'}`,
                            color: inWatchlist ? '#e50914' : '#a1a1aa',
                        }}
                    >
                        <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor' className='w-4 h-4'><path d='M6.3 2.84A1.5 1.5 0 0 0 5 4.312v11.376a.5.5 0 0 0 .77.419l4.23-2.791 4.23 2.79a.5.5 0 0 0 .77-.418V4.313a1.5 1.5 0 0 0-1.3-1.472A42.5 42.5 0 0 0 10 2.5a42.5 42.5 0 0 0-3.7.34Z' /></svg>
                        {inWatchlist ? 'In Watchlist' : 'Add to Watchlist'}
                    </button>
                </div>
            </div>

            {/* ── Write review ─────────────────────────────────────────────── */}
            <div className={sectionBox} style={sectionStyle}>
                <h2 className='text-xs uppercase tracking-widest text-zinc-500 mb-4'>Your review</h2>
                <textarea
                    value={review}
                    onChange={e => setReview(e.target.value)}
                    placeholder='Share your thoughts about this movie…'
                    rows={4}
                    className='w-full text-sm text-zinc-200 rounded-xl px-4 py-3 resize-none outline-none focus:ring-1 placeholder-zinc-600'
                    style={{
                        background: 'rgba(255,255,255,0.05)',
                        border: '1px solid rgba(255,255,255,0.10)',
                        // @ts-ignore
                        '--tw-ring-color': 'var(--netflix-red)',
                    }}
                />
                <div className='flex items-center gap-3 mt-3'>
                    <button
                        onClick={handleSave}
                        disabled={userRating === 0 && review.trim() === ''}
                        className='px-6 py-2 rounded-full font-semibold text-sm text-white transition-opacity disabled:opacity-40 disabled:cursor-not-allowed'
                        style={{ background: 'var(--netflix-red)' }}
                    >
                        Save
                    </button>
                    {saved && <span className='text-sm text-green-400 animate-pulse'>Saved!</span>}
                </div>
            </div>

            {/* ── Community reviews ────────────────────────────────────────── */}
            <div>
                <h2 className='text-lg font-bold text-white mb-4'>
                    Reviews
                    {savedReviews.length > 0 && (
                        <span className='ml-2 text-sm font-normal text-zinc-500'>({savedReviews.length})</span>
                    )}
                </h2>
                {savedReviews.length === 0 ? (
                    <div
                        className='rounded-2xl px-6 py-8 text-center text-zinc-500 text-sm'
                        style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}
                    >
                        No reviews yet — be the first to rate this movie.
                    </div>
                ) : (
                    <div className='flex flex-col gap-4'>
                        {savedReviews.map((r, i) => (
                            <div
                                key={i}
                                className='rounded-2xl px-6 py-4 flex flex-col gap-2'
                                style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
                            >
                                <div className='flex items-center justify-between gap-4'>
                                    <div className='flex items-center gap-3'>
                                        <div
                                            className='w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white'
                                            style={{ background: `hsl(${(movie.id * 17 + i * 73) % 360}, 50%, 40%)` }}
                                        >
                                            U
                                        </div>
                                        <StarRating value={r.rating} size='sm' />
                                    </div>
                                    <span className='text-xs text-zinc-500'>
                                        {new Date(r.savedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                                    </span>
                                </div>
                                {r.review.trim() && (
                                    <p className='text-sm text-zinc-300 leading-relaxed pl-11'>{r.review}</p>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* ── Similar movies ───────────────────────────────────────────── */}
            {similar.length > 0 && (
                <div>
                    <h2 className='text-lg font-bold text-white mb-4'>Similar movies</h2>
                    <div className='relative'>
                        {/* Left arrow */}
                        <button
                            onClick={() => scrollBy(-1)}
                            className='absolute left-0 top-1/2 -translate-y-1/2 z-10 w-9 h-9 flex items-center justify-center rounded-full text-white -ml-4'
                            style={{ background: 'rgba(0,0,0,0.7)', border: '1px solid rgba(255,255,255,0.12)' }}
                        >
                            <svg xmlns='http://www.w3.org/2000/svg' className='w-4 h-4' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={2.5}>
                                <path strokeLinecap='round' strokeLinejoin='round' d='M15 19l-7-7 7-7' />
                            </svg>
                        </button>

                        <div
                            ref={scrollRef}
                            className='flex gap-4 overflow-x-auto pb-2 scrollbar-thin'
                            style={{ scrollbarWidth: 'none' }}
                        >
                            {similar.map(m => (
                                <SimilarMovieCard key={m.id} movie={m} />
                            ))}
                        </div>

                        {/* Right arrow */}
                        <button
                            onClick={() => scrollBy(1)}
                            className='absolute right-0 top-1/2 -translate-y-1/2 z-10 w-9 h-9 flex items-center justify-center rounded-full text-white -mr-4'
                            style={{ background: 'rgba(0,0,0,0.7)', border: '1px solid rgba(255,255,255,0.12)' }}
                        >
                            <svg xmlns='http://www.w3.org/2000/svg' className='w-4 h-4' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={2.5}>
                                <path strokeLinecap='round' strokeLinejoin='round' d='M9 5l7 7-7 7' />
                            </svg>
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}
