'use client'

import { fetchAllMovies } from '@/lib/api'
import type { Movie } from '@/lib/api'
import {
	getAllReviews,
	getGenreStats,
	getWatchlist,
	getWatched,
	removeFromWatchlist,
	removeWatched,
	type UserReview,
	type WatchedEntry,
	type WatchlistEntry,
} from '@/lib/userStore'
import { useEffect, useRef, useState } from 'react'

// ── Mini poster card used inside the profile lists ───────────────────────────
const CARD_GRADIENTS = [
	'135deg, #0f2027 0%, #203a43 50%, #2c5364 100%',
	'135deg, #200122 0%, #6f0000 100%',
	'135deg, #0a3d0c 0%, #0b6e4f 100%',
	'135deg, #2d1b69 0%, #11998e 100%',
	'135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%',
	'135deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%',
	'135deg, #134e5e 0%, #71b280 100%',
	'135deg, #373b44 0%, #4286f4 100%',
]

function MiniCard({
	movie,
	badge,
	onRemove,
}: {
	movie: Movie
	badge?: React.ReactNode
	onRemove?: () => void
}) {
	const gradient = CARD_GRADIENTS[movie.id % CARD_GRADIENTS.length]
	const [poster, setPoster] = useState<string | null>(null)

	useEffect(() => {
		let cancelled = false
		import('@/lib/api').then(({ fetchMovieDetails }) =>
			fetchMovieDetails(movie.id).then(d => {
				if (!cancelled) setPoster(d?.poster_url ?? null)
			})
		)
		return () => { cancelled = true }
	}, [movie.id])

	return (
		<div
			className='relative rounded-xl overflow-hidden flex-shrink-0 group'
			style={{
				width: 140,
				border: '1px solid rgba(255,255,255,0.09)',
				boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
				background: 'var(--bg-card)',
			}}
		>
			<div
				className='relative overflow-hidden'
				style={{ aspectRatio: '2/3', background: `linear-gradient(${gradient})` }}
			>
				{poster && (
					<img src={poster} alt={movie.title} className='absolute inset-0 w-full h-full object-cover' />
				)}
				{badge && (
					<div className='absolute top-2 left-2 z-10'>{badge}</div>
				)}
				{onRemove && (
					<button
						onClick={onRemove}
						className='absolute top-2 right-2 z-10 w-6 h-6 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity'
						style={{ background: 'rgba(0,0,0,0.75)' }}
					>
						<svg className='w-3.5 h-3.5 text-zinc-300' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={2.5}>
							<path strokeLinecap='round' strokeLinejoin='round' d='M6 18L18 6M6 6l12 12' />
						</svg>
					</button>
				)}
			</div>
			<div className='px-2.5 py-2' style={{ background: '#1f1f1f', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
				<p className='text-xs font-semibold text-white line-clamp-1'>{movie.title}</p>
				{movie.year && <p className='text-xs text-zinc-600 mt-0.5'>{movie.year}</p>}
			</div>
		</div>
	)
}

// ── Horizontal scrollable row ────────────────────────────────────────────────
function MovieRow({ children, empty }: { children: React.ReactNode; empty: React.ReactNode }) {
	const ref = useRef<HTMLDivElement>(null)
	const hasChildren = (children as React.ReactNode[])?.length > 0

	if (!hasChildren) return (
		<div
			className='flex flex-col items-center justify-center py-12 rounded-2xl gap-3'
			style={{ background: 'rgba(255,255,255,0.02)', border: '1px dashed rgba(255,255,255,0.08)' }}
		>
			{empty}
		</div>
	)

	return (
		<div
			ref={ref}
			className='flex gap-4 overflow-x-auto pb-2 scrollbar-hide'
		>
			{children}
		</div>
	)
}

// ── Star display (read-only) ─────────────────────────────────────────────────
function Stars({ value }: { value: number }) {
	return (
		<span className='flex gap-0.5'>
			{[1, 2, 3, 4, 5].map(s => (
				<span key={s} style={{ color: s <= value ? '#f59e0b' : 'rgba(255,255,255,0.15)', fontSize: 14 }}>★</span>
			))}
		</span>
	)
}

// ── Genre bar chart ──────────────────────────────────────────────────────────
function GenreChart({ stats }: { stats: { genre: string; count: number }[] }) {
	if (!stats.length) return null
	const max = stats[0].count
	const BAR_COLORS = [
		'#e50914', '#e87c1e', '#f0c040', '#46d369',
		'#0080ff', '#b44be1', '#00c9c9', '#ff6b9d',
	]
	return (
		<div className='flex flex-col gap-2.5'>
			{stats.map(({ genre, count }, i) => (
				<div key={genre} className='flex items-center gap-3'>
					<span className='text-xs text-zinc-400 w-28 flex-shrink-0 text-right truncate'>{genre}</span>
					<div className='flex-1 h-5 rounded-full overflow-hidden' style={{ background: 'rgba(255,255,255,0.06)' }}>
						<div
							className='h-full rounded-full transition-all duration-700'
							style={{
								width: `${(count / max) * 100}%`,
								background: BAR_COLORS[i % BAR_COLORS.length],
								opacity: 0.85,
							}}
						/>
					</div>
					<span className='text-xs text-zinc-500 w-6 text-right flex-shrink-0'>{count}</span>
				</div>
			))}
		</div>
	)
}

// ── Review poster (loads real poster, falls back to gradient) ──────────────
function ReviewPoster({ movieId, gradient }: { movieId: number; gradient: string }) {
	const [poster, setPoster] = useState<string | null>(null)

	useEffect(() => {
		let cancelled = false
		import('@/lib/api').then(({ fetchMovieDetails }) =>
			fetchMovieDetails(movieId).then(d => {
				if (!cancelled) setPoster(d?.poster_url ?? null)
			})
		)
		return () => { cancelled = true }
	}, [movieId])

	return (
		<div
			className='relative rounded-lg overflow-hidden flex-shrink-0'
			style={{ width: 52, aspectRatio: '2/3', background: `linear-gradient(${gradient})` }}
		>
			{poster && (
				<img src={poster} alt='' className='absolute inset-0 w-full h-full object-cover' />
			)}
		</div>
	)
}

// ── Section wrapper ──────────────────────────────────────────────────────────
function Section({
	title,
	icon,
	count,
	children,
}: {
	title: string
	icon: React.ReactNode
	count?: number
	children: React.ReactNode
}) {
	return (
		<div
			className='rounded-2xl p-6 md:p-8'
			style={{
				background: 'var(--bg-card)',
				border: '1px solid rgba(255,255,255,0.07)',
			}}
		>
			<div className='flex items-center gap-3 mb-6'>
				<span className='text-zinc-400'>{icon}</span>
				<h2 className='text-lg font-bold text-white'>{title}</h2>
				{count !== undefined && (
					<span
						className='ml-auto text-xs font-bold px-2.5 py-1 rounded-full'
						style={{ background: 'rgba(255,255,255,0.07)', color: '#71717a' }}
					>
						{count}
					</span>
				)}
			</div>
			{children}
		</div>
	)
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function ProfilePage() {
	const [movies, setMovies]       = useState<Record<number, Movie>>({})
	const [watched, setWatched]     = useState<WatchedEntry[]>([])
	const [watchlist, setWatchlist] = useState<WatchlistEntry[]>([])
	const [reviews, setReviews]     = useState<UserReview[]>([])
	const [ready, setReady]         = useState(false)

	// Load everything from localStorage + fetch movie metadata
	useEffect(() => {
		const w  = getWatched()
		const wl = getWatchlist()
		const rv = getAllReviews()

		setWatched(w)
		setWatchlist(wl)
		setReviews(rv)

		// Collect all unique IDs we need
		const ids = new Set([
			...w.map(e => e.movieId),
			...wl.map(e => e.movieId),
			...rv.map(e => e.movieId),
		])

		if (ids.size === 0) { setReady(true); return }

		// Fetch catalog once and build lookup map
		fetchAllMovies().then(all => {
			const map: Record<number, Movie> = {}
			all.forEach(m => { if (ids.has(m.id)) map[m.id] = m })
			setMovies(map)
			setReady(true)
		})
	}, [])

	function handleRemoveWatched(id: number) {
		removeWatched(id)
		setWatched(prev => prev.filter(e => e.movieId !== id))
	}

	function handleRemoveWatchlist(id: number) {
		removeFromWatchlist(id)
		setWatchlist(prev => prev.filter(e => e.movieId !== id))
	}

	// Stats
	const genreMap: Record<number, string | null> = {}
	Object.values(movies).forEach(m => { genreMap[m.id] = m.genres })
	const genreStats = getGenreStats(genreMap)

	const totalRated   = reviews.filter(r => r.rating > 0).length
	const avgRating    = totalRated
		? (reviews.filter(r => r.rating > 0).reduce((s, r) => s + r.rating, 0) / totalRated).toFixed(1)
		: '—'
	const totalReviews = reviews.filter(r => r.review.trim()).length

	if (!ready) return (
		<div className='min-h-screen flex items-center justify-center' style={{ background: 'var(--bg-primary)' }}>
			<div className='w-8 h-8 rounded-full border-2 border-red-600 border-t-transparent animate-spin' />
		</div>
	);

	return (
		<div className='min-h-screen pt-24 pb-20 px-4 md:px-8' style={{ background: 'var(--bg-primary)' }}>
			{/* Top glow */}
			<div
				className='fixed top-0 left-0 right-0 h-72 pointer-events-none'
				style={{ background: 'linear-gradient(to bottom, rgba(229,9,20,0.04) 0%, transparent 100%)' }}
			/>

			<div className='relative max-w-5xl mx-auto flex flex-col gap-6'>

				{/* ── Stats bar ──────────────────────────────────────────── */}
				<div className='grid grid-cols-2 sm:grid-cols-4 gap-4'>
					{[
						{
							icon: (
								<svg className='w-6 h-6' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.6}><path strokeLinecap='round' strokeLinejoin='round' d='M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z' /></svg>
							),
							label: 'Watched', value: watched.length,
						},
						{
							icon: (
								<svg className='w-6 h-6' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.6}><path strokeLinecap='round' strokeLinejoin='round' d='M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0 1 11.186 0Z' /></svg>
							),
							label: 'Watchlist', value: watchlist.length,
						},
						{
							icon: (
								<svg className='w-6 h-6' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.6}><path strokeLinecap='round' strokeLinejoin='round' d='M11.48 3.499a.562.562 0 0 1 1.04 0l2.125 5.111a.563.563 0 0 0 .475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 0 0-.182.557l1.285 5.385a.562.562 0 0 1-.84.61l-4.725-2.885a.562.562 0 0 0-.586 0L6.982 20.54a.562.562 0 0 1-.84-.61l1.285-5.386a.562.562 0 0 0-.182-.557l-4.204-3.602a.562.562 0 0 1 .321-.988l5.518-.442a.563.563 0 0 0 .475-.345L11.48 3.5Z' /></svg>
							),
							label: 'Rated', value: totalRated,
						},
						{
							icon: (
								<svg className='w-6 h-6' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.6}><path strokeLinecap='round' strokeLinejoin='round' d='M16.862 4.487l1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125' /></svg>
							),
							label: 'Reviews', value: totalReviews,
						},
					].map(s => (
						<div
							key={s.label}
							className='rounded-2xl p-5 flex flex-col items-center gap-1'
							style={{ background: 'var(--bg-card)', border: '1px solid rgba(255,255,255,0.07)' }}
						>
							<span className='text-zinc-500'>{s.icon}</span>
							<span className='text-2xl font-black text-white'>{s.value}</span>
							<span className='text-xs text-zinc-500'>{s.label}</span>
						</div>
					))}
				</div>

				{/* ── Watched ────────────────────────────────────────────── */}
				<Section
					title='Watched'
					icon={<svg className='w-5 h-5' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.8}><path strokeLinecap='round' strokeLinejoin='round' d='M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z' /></svg>}
					count={watched.length}
				>
					<MovieRow
						empty={
							<>
								<svg className='w-10 h-10 text-zinc-700' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.4}><path strokeLinecap='round' strokeLinejoin='round' d='M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z' /></svg>
								<p className='text-zinc-400 text-sm'>No watched movies yet</p>
								<p className='text-zinc-600 text-xs'>Open any movie and click "Watched"</p>
							</>
						}
					>
						{watched.map(({ movieId }) => {
							const m = movies[movieId]
							if (!m) return null
							return (
								<MiniCard
									key={movieId}
									movie={m}
									badge={
										<span className='text-xs font-bold px-1.5 py-0.5 rounded-md' style={{ background: 'rgba(46,160,67,0.85)', color: '#fff' }}>
											✓
										</span>
									}
									onRemove={() => handleRemoveWatched(movieId)}
								/>
							)
						})}
					</MovieRow>
				</Section>

				{/* ── Watchlist ──────────────────────────────────────────── */}
				<Section
					title='Watchlist'
					icon={<svg className='w-5 h-5' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.8}><path strokeLinecap='round' strokeLinejoin='round' d='M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0 1 11.186 0Z' /></svg>}
					count={watchlist.length}
				>
					<MovieRow
						empty={
							<>
								<svg className='w-10 h-10 text-zinc-700' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.4}><path strokeLinecap='round' strokeLinejoin='round' d='M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0 1 11.186 0Z' /></svg>
								<p className='text-zinc-400 text-sm'>Your watchlist is empty</p>
								<p className='text-zinc-600 text-xs'>Open any movie and click "Watchlist"</p>
							</>
						}
					>
						{watchlist.map(({ movieId }) => {
							const m = movies[movieId]
							if (!m) return null
							return (
								<MiniCard
									key={movieId}
									movie={m}
									onRemove={() => handleRemoveWatchlist(movieId)}
								/>
							)
						})}
					</MovieRow>
				</Section>

				{/* ── Ratings & Reviews ──────────────────────────────────── */}
				<Section
					title='Ratings & Reviews'
					icon={<svg className='w-5 h-5' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.8}><path strokeLinecap='round' strokeLinejoin='round' d='M11.48 3.499a.562.562 0 0 1 1.04 0l2.125 5.111a.563.563 0 0 0 .475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 0 0-.182.557l1.285 5.385a.562.562 0 0 1-.84.61l-4.725-2.885a.562.562 0 0 0-.586 0L6.982 20.54a.562.562 0 0 1-.84-.61l1.285-5.386a.562.562 0 0 0-.182-.557l-4.204-3.602a.562.562 0 0 1 .321-.988l5.518-.442a.563.563 0 0 0 .475-.345L11.48 3.5Z' /></svg>}
					count={reviews.length}
				>
					{reviews.length === 0 ? (
						<div
							className='flex flex-col items-center justify-center py-12 rounded-2xl gap-3'
							style={{ background: 'rgba(255,255,255,0.02)', border: '1px dashed rgba(255,255,255,0.08)' }}
						>
							<svg className='w-10 h-10 text-zinc-700' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.4}><path strokeLinecap='round' strokeLinejoin='round' d='M16.862 4.487l1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125' /></svg>
							<p className='text-zinc-400 text-sm'>No ratings yet</p>
							<p className='text-zinc-600 text-xs'>Rate movies to see them here</p>
						</div>
					) : (
						<div className='flex flex-col gap-4'>
							{/* Average rating summary */}
							<div
								className='flex items-center gap-4 px-5 py-4 rounded-xl'
								style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)' }}
							>
								<div className='text-center'>
									<div className='text-3xl font-black text-white'>{avgRating}</div>
									<div className='text-xs text-zinc-500 mt-0.5'>avg rating</div>
								</div>
								<div className='w-px h-10' style={{ background: 'rgba(255,255,255,0.1)' }} />
								<div className='flex flex-col gap-1 flex-1'>
									{[5, 4, 3, 2, 1].map(star => {
										const cnt = reviews.filter(r => r.rating === star).length
										return (
											<div key={star} className='flex items-center gap-2'>
												<span className='text-xs text-zinc-500 w-4'>{star}★</span>
												<div className='flex-1 h-1.5 rounded-full overflow-hidden' style={{ background: 'rgba(255,255,255,0.06)' }}>
													<div
														className='h-full rounded-full'
														style={{
															width: totalRated ? `${(cnt / totalRated) * 100}%` : '0%',
															background: '#f59e0b',
														}}
													/>
												</div>
												<span className='text-xs text-zinc-600 w-4 text-right'>{cnt}</span>
											</div>
										)
									})}
								</div>
							</div>

							{/* Review cards */}
							<div className='flex flex-col gap-3'>
								{reviews.map(rv => {
									const m = movies[rv.movieId]
									const gradient = CARD_GRADIENTS[rv.movieId % CARD_GRADIENTS.length]
									return (
										<div
											key={rv.movieId}
											className='flex gap-4 rounded-xl p-4'
											style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
										>
											{/* Mini poster */}
											<ReviewPoster movieId={rv.movieId} gradient={gradient} />
											<div className='flex flex-col gap-1.5 flex-1 min-w-0'>
												<div className='flex items-start justify-between gap-2'>
													<p className='text-sm font-semibold text-white truncate'>
														{m?.title ?? `Movie #${rv.movieId}`}
													</p>
													{rv.savedAt && (
														<span className='text-xs text-zinc-600 flex-shrink-0'>
															{new Date(rv.savedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
														</span>
													)}
												</div>
												{rv.rating > 0 && <Stars value={rv.rating} />}
												{rv.review.trim() && (
													<p className='text-sm text-zinc-400 leading-relaxed line-clamp-3'>{rv.review}</p>
												)}
											</div>
										</div>
									)
								})}
							</div>
						</div>
					)}
				</Section>

				{/* ── Genre Stats ────────────────────────────────────────── */}
				{genreStats.length > 0 && (
					<Section
						title='Favourite Genres'
						icon={<svg className='w-5 h-5' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.8}><path strokeLinecap='round' strokeLinejoin='round' d='M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z' /></svg>}
					>
						<div className='flex flex-col md:flex-row gap-8 items-start'>
							{/* Bar chart */}
							<div className='flex-1 w-full'>
								<GenreChart stats={genreStats} />
							</div>
							{/* Donut-style legend */}
							<div className='flex flex-col gap-2 flex-shrink-0'>
								{genreStats.slice(0, 5).map(({ genre, count }, i) => {
									const COLORS = ['#e50914', '#e87c1e', '#f0c040', '#46d369', '#0080ff']
									return (
										<div key={genre} className='flex items-center gap-2'>
											<div className='w-2.5 h-2.5 rounded-full flex-shrink-0' style={{ background: COLORS[i] }} />
											<span className='text-sm text-zinc-300'>{genre}</span>
											<span className='text-xs text-zinc-600 ml-auto pl-4'>{count}×</span>
										</div>
									)
								})}
							</div>
						</div>
					</Section>
				)}

			</div>
		</div>
	)
}
