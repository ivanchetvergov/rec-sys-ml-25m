'use client'

import type { Movie, MovieDetails } from '@/lib/api'
import {
	addWatchedDB,
	addToWatchlist as apiAddWatchlist,
	removeFromWatchlist as apiRemoveWatchlist,
	fetchMovieDetails,
	removeWatchedDB,
	upsertReview,
} from '@/lib/api'
import { getToken } from '@/lib/authStore'
import { useCallback, useEffect, useState } from 'react'

interface Props {
	movie: Movie
	onClose: () => void
}

const STAR_COUNT = 5

function StarRating({
	value,
	onChange,
}: {
	value: number
	onChange: (v: number) => void
}) {
	const [hovered, setHovered] = useState(0)
	return (
		<div className='flex gap-1'>
			{Array.from({ length: STAR_COUNT }, (_, i) => i + 1).map(star => (
				<button
					key={star}
					type='button'
					onMouseEnter={() => setHovered(star)}
					onMouseLeave={() => setHovered(0)}
					onClick={() => onChange(star)}
					aria-label={`Rate ${star} out of ${STAR_COUNT}`}
					className='text-2xl transition-transform hover:scale-110 focus:outline-none'
					style={{
						color:
							star <= (hovered || value) ? '#f59e0b' : 'rgba(255,255,255,0.2)',
					}}
				>
					★
				</button>
			))}
		</div>
	)
}

const pillBase: React.CSSProperties = {
	background: 'rgba(255,255,255,0.08)',
	border: '1px solid rgba(255,255,255,0.15)',
	color: '#d4d4d8',
}
const pillClass =
	'flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold transition-opacity hover:opacity-80'

export function MovieDetailModal({ movie, onClose }: Props) {
	const [details, setDetails] = useState<MovieDetails | null>(null)
	const [loading, setLoading] = useState(true)
	const [userRating, setUserRating] = useState(0)
	const [review, setReview] = useState('')
	const [saved, setSaved] = useState(false)
	const [watched, setWatched] = useState(false)
	const [inWatchlist, setInWatchlist] = useState(false)

	// Load TMDB details
	useEffect(() => {
		setLoading(true)
		fetchMovieDetails(movie.id)
			.then(setDetails)
			.finally(() => setLoading(false))
	}, [movie.id])

	// Load saved review + watchlist + watched state from DB (if logged in)
	useEffect(() => {
		const token = getToken()
		if (!token) return
		import('@/lib/api').then(
			({ fetchReviews, fetchWatchlist, fetchWatched }) => {
				fetchReviews(token).then(all => {
					const mine = all.find(r => r.movie_id === movie.id)
					if (mine) {
						setUserRating(mine.rating)
						setReview(mine.review_text ?? '')
					}
				})
				fetchWatchlist(token).then(wl => {
					setInWatchlist(wl.some(w => w.movie_id === movie.id))
				})
				fetchWatched(token).then(wl => {
					setWatched(wl.some(w => w.movie_id === movie.id))
				})
			},
		)
	}, [movie.id])

	// Close on Escape
	const handleKeyDown = useCallback(
		(e: KeyboardEvent) => {
			if (e.key === 'Escape') onClose()
		},
		[onClose],
	)
	useEffect(() => {
		document.addEventListener('keydown', handleKeyDown)
		return () => document.removeEventListener('keydown', handleKeyDown)
	}, [handleKeyDown])

	const handleSave = async () => {
		const token = getToken()
		if (!token) return
		await upsertReview(token, movie.id, movie.title, userRating, review)
		setSaved(true)
		setTimeout(() => setSaved(false), 2000)
	}

	const genres = movie.genres?.split('|') ?? []
	const posterUrl = details?.poster_url
	const gradient = `linear-gradient(135deg, hsl(${(movie.id * 37) % 360},40%,25%), hsl(${(movie.id * 37 + 120) % 360},35%,15%))`

	return (
		/* Backdrop */
		<div
			className='fixed inset-0 z-50 flex items-center justify-center p-4'
			style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(4px)' }}
			onClick={onClose}
		>
			{/* Panel — overflow-hidden so poster fills height via flex stretch */}
			<div
				className='relative w-full max-w-3xl max-h-[90vh] rounded-xl shadow-2xl overflow-hidden'
				style={{ background: '#181818' }}
				onClick={e => e.stopPropagation()}
			>
				{/* Close button */}
				<button
					onClick={onClose}
					className='absolute top-4 right-4 z-10 w-8 h-8 flex items-center justify-center rounded-full text-white'
					style={{ background: 'rgba(0,0,0,0.6)' }}
					aria-label='Close'
				>
					<svg
						xmlns='http://www.w3.org/2000/svg'
						className='w-5 h-5'
						fill='none'
						viewBox='0 0 24 24'
						stroke='currentColor'
						strokeWidth={2.5}
					>
						<path
							strokeLinecap='round'
							strokeLinejoin='round'
							d='M6 18L18 6M6 6l12 12'
						/>
					</svg>
				</button>

				{/* flex-row: poster stretches to content height automatically */}
				<div className='flex flex-row'>
					{/* Poster — full height (self-stretch via flex default align-items:stretch) */}
					<div className='w-80 flex-shrink-0 hidden md:block'>
						<div
							className='relative h-full overflow-hidden rounded-l-xl'
							style={{ background: gradient }}
						>
							{loading && (
								<div
									className='absolute inset-0 animate-pulse'
									style={{ background: 'rgba(255,255,255,0.05)' }}
								/>
							)}
							{posterUrl && (
								<img
									src={posterUrl}
									alt={movie.title}
									className='absolute inset-0 w-full h-full object-cover'
								/>
							)}
							{!posterUrl && !loading && (
								<div className='absolute inset-0 flex items-end pb-6 px-4'>
									<span className='text-white font-bold text-base leading-snug drop-shadow-lg'>
										{movie.title}
									</span>
								</div>
							)}
						</div>
					</div>

					{/* Content — scrollable */}
					<div className='flex-1 p-6 flex flex-col gap-4 overflow-y-auto max-h-[90vh]'>
						{/* Title & meta */}
						<div>
							<h2 className='text-2xl font-bold text-white leading-tight'>
								{movie.title}
							</h2>
							{details?.tagline && (
								<p
									className='text-sm italic mt-1'
									style={{ color: 'var(--netflix-red)' }}
								>
									"{details.tagline}"
								</p>
							)}
							<div className='flex flex-wrap gap-3 mt-2 text-sm text-zinc-400'>
								{movie.year && <span>{movie.year}</span>}
								{details?.runtime && <span>{details.runtime} min</span>}
								{details?.release_date && (
									<span>{details.release_date.slice(0, 4)}</span>
								)}
							</div>
						</div>

						{/* Genres */}
						{genres.length > 0 && (
							<div className='flex flex-wrap gap-2'>
								{genres.map(g => (
									<span
										key={g}
										className='text-xs px-2.5 py-1 rounded-full font-medium'
										style={{
											background: 'rgba(255,255,255,0.08)',
											color: '#d4d4d8',
										}}
									>
										{g}
									</span>
								))}
							</div>
						)}

						{/* Overview */}
						{details?.overview && (
							<p className='text-sm leading-relaxed text-zinc-300'>
								{details.overview}
							</p>
						)}

						{/* Ratings row */}
						<div className='flex flex-wrap gap-5 text-sm'>
							{movie.avg_rating != null && (
								<div className='flex flex-col'>
									<span className='text-zinc-500 text-xs uppercase tracking-wide'>
										Avg rating
									</span>
									<span className='text-white font-bold text-lg'>
										{movie.avg_rating.toFixed(1)}
									</span>
									<span className='text-zinc-500 text-xs'>
										{movie.num_ratings?.toLocaleString()} reviews
									</span>
								</div>
							)}
							{details?.tmdb_rating != null && (
								<div className='flex flex-col'>
									<span className='text-zinc-500 text-xs uppercase tracking-wide'>
										TMDB
									</span>
									<span className='text-white font-bold text-lg'>
										{details.tmdb_rating.toFixed(1)}
									</span>
									<span className='text-zinc-500 text-xs'>
										{details.tmdb_votes?.toLocaleString()} votes
									</span>
								</div>
							)}
						</div>

						{/* Divider */}
						<div style={{ height: 1, background: 'rgba(255,255,255,0.08)' }} />

						{/* Your rating + Watched + Watchlist — all on one row */}
						<div>
							<p className='text-xs uppercase tracking-wide text-zinc-500 mb-2'>
								Your rating
							</p>
							<div className='flex items-center gap-3 flex-wrap'>
								<StarRating value={userRating} onChange={setUserRating} />
								{/* Watched */}
								<button
									onClick={async () => {
										const token = getToken()
										if (!token) return
										if (watched) {
											await removeWatchedDB(token, movie.id)
											setWatched(false)
										} else {
											await addWatchedDB(token, movie)
											setWatched(true)
										}
									}}
									className='flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all'
									style={{
										background: watched
											? 'rgba(46,160,67,0.2)'
											: 'rgba(255,255,255,0.07)',
										border: `1px solid ${watched ? 'rgba(46,160,67,0.5)' : 'rgba(255,255,255,0.12)'}`,
										color: watched ? '#4ade80' : '#a1a1aa',
									}}
								>
									{watched ? (
										<svg
											xmlns='http://www.w3.org/2000/svg'
											viewBox='0 0 20 20'
											fill='currentColor'
											className='w-3.5 h-3.5'
										>
											<path
												fillRule='evenodd'
												d='M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z'
												clipRule='evenodd'
											/>
										</svg>
									) : (
										<svg
											xmlns='http://www.w3.org/2000/svg'
											viewBox='0 0 20 20'
											fill='currentColor'
											className='w-3.5 h-3.5'
										>
											<path d='M10.75 4.75a.75.75 0 0 0-1.5 0v4.5h-4.5a.75.75 0 0 0 0 1.5h4.5v4.5a.75.75 0 0 0 1.5 0v-4.5h4.5a.75.75 0 0 0 0-1.5h-4.5v-4.5Z' />
										</svg>
									)}{' '}
									Watched
								</button>
								{/* Watchlist */}
								<button
									onClick={async () => {
										const token = getToken()
										if (!token) return
										if (inWatchlist) {
											await apiRemoveWatchlist(token, movie.id)
											setInWatchlist(false)
										} else {
											await apiAddWatchlist(token, movie)
											setInWatchlist(true)
										}
									}}
									className='flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all'
									style={{
										background: inWatchlist
											? 'rgba(229,9,20,0.15)'
											: 'rgba(255,255,255,0.07)',
										border: `1px solid ${inWatchlist ? 'rgba(229,9,20,0.4)' : 'rgba(255,255,255,0.12)'}`,
										color: inWatchlist ? '#e50914' : '#a1a1aa',
									}}
								>
									<svg
										xmlns='http://www.w3.org/2000/svg'
										viewBox='0 0 20 20'
										fill='currentColor'
										className='w-3.5 h-3.5'
									>
										<path d='M6.3 2.84A1.5 1.5 0 0 0 5 4.312v11.376a.5.5 0 0 0 .77.419l4.23-2.791 4.23 2.79a.5.5 0 0 0 .77-.418V4.313a1.5 1.5 0 0 0-1.3-1.472A42.5 42.5 0 0 0 10 2.5a42.5 42.5 0 0 0-3.7.34Z' />
									</svg>
									{inWatchlist ? 'In Watchlist' : 'Watchlist'}
								</button>
							</div>
						</div>

						{/* Review textarea */}
						<div>
							<p className='text-xs uppercase tracking-wide text-zinc-500 mb-2'>
								Your review
							</p>
							<textarea
								value={review}
								onChange={e => setReview(e.target.value)}
								placeholder='Write a few words about the movie...'
								rows={3}
								className='w-full text-sm text-zinc-300 rounded-lg px-3 py-2.5 resize-none outline-none focus:ring-1 placeholder-zinc-600'
								style={{
									background: 'rgba(255,255,255,0.06)',
									border: '1px solid rgba(255,255,255,0.1)',
									// @ts-ignore
									'--tw-ring-color': 'var(--netflix-red)',
								}}
							/>
						</div>

						{/* Save left | IMDB + TMDB + Full page right */}
						<div className='flex items-center justify-between gap-2 mt-auto pt-1'>
							{/* Left: Save */}
							<div className='flex items-center gap-2'>
								<button
									onClick={handleSave}
									disabled={userRating === 0 && review.trim() === ''}
									className='px-4 py-1.5 rounded-full font-semibold text-sm text-white transition-opacity disabled:opacity-40 disabled:cursor-not-allowed'
									style={{ background: 'var(--netflix-red)' }}
								>
									Save
								</button>
								{saved && (
									<span className='text-sm text-green-400 animate-pulse'>
										Saved!
									</span>
								)}
							</div>
							{/* Right: external links */}
							<div className='flex items-center gap-2'>
								{movie.imdb_id && (
									<a
										href={`https://www.imdb.com/title/tt${String(movie.imdb_id).padStart(7, '0')}/`}
										target='_blank'
										rel='noopener noreferrer'
										className={pillClass}
										style={pillBase}
									>
										<svg
											xmlns='http://www.w3.org/2000/svg'
											className='w-3.5 h-3.5'
											fill='none'
											viewBox='0 0 24 24'
											stroke='currentColor'
											strokeWidth={2}
										>
											<path
												strokeLinecap='round'
												strokeLinejoin='round'
												d='M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14'
											/>
										</svg>
										IMDB
									</a>
								)}
								{movie.tmdb_id && (
									<a
										href={`https://www.themoviedb.org/movie/${movie.tmdb_id}`}
										target='_blank'
										rel='noopener noreferrer'
										className={pillClass}
										style={pillBase}
									>
										<svg
											xmlns='http://www.w3.org/2000/svg'
											className='w-3.5 h-3.5'
											fill='none'
											viewBox='0 0 24 24'
											stroke='currentColor'
											strokeWidth={2}
										>
											<path
												strokeLinecap='round'
												strokeLinejoin='round'
												d='M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14'
											/>
										</svg>
										TMDB
									</a>
								)}
								<a
									href={`/movies/${movie.id}`}
									target='_blank'
									rel='noopener noreferrer'
									className={pillClass}
									style={pillBase}
								>
									<svg
										xmlns='http://www.w3.org/2000/svg'
										className='w-3.5 h-3.5'
										fill='none'
										viewBox='0 0 24 24'
										stroke='currentColor'
										strokeWidth={2}
									>
										<path
											strokeLinecap='round'
											strokeLinejoin='round'
											d='M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14'
										/>
									</svg>
									Full page
								</a>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	)
}
