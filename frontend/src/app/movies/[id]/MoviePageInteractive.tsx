'use client'

/**
 * MoviePageInteractive — client-side interactive section for the movie detail page.
 *
 * Includes:
 *  - Star rating + Watched + Watchlist buttons (DB-backed)
 *  - Write/edit review textarea with Save (DB-backed)
 *  - Similar movies row (ALS cosine similarity via /api/movies/{id}/similar)
 */
import type { Movie, PublicMovieReview } from '@/lib/api'
import {
	addWatchedDB,
	addToWatchlist as apiAddWatchlist,
	removeFromWatchlist as apiRemoveWatchlist,
	fetchMovieReviews,
	fetchReviews,
	fetchSimilarMovies,
	fetchWatched,
	fetchWatchlist,
	removeWatchedDB,
	upsertReview,
} from '@/lib/api'
import {
	isLoggedIn as checkIsLoggedIn,
	getAuthUser,
	getToken,
} from '@/lib/authStore'
import { AvatarIcon } from '@/lib/avatars'
import { trackKpi } from '@/lib/kpi'
import { useTranslation } from '@/lib/useTranslation'
import Link from 'next/link'
import { useEffect, useRef, useState } from 'react'

// ── Types ────────────────────────────────────────────────────────────────────
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
	const sizeClass =
		size === 'lg' ? 'text-3xl' : size === 'sm' ? 'text-lg' : 'text-2xl'
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
					style={{
						color:
							star <= (hovered || value) ? '#f59e0b' : 'rgba(255,255,255,0.18)',
					}}
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
			.then(r => (r.ok ? r.json() : null))
			.then(d => d?.poster_url && setPosterUrl(d.poster_url))
			.catch(() => { })
	}, [movie.id, movie.tmdb_id])

	return (
		<Link
			href={`/movies/${movie.id}`}
			className='flex-shrink-0 w-36 group cursor-pointer'
		>
			<div
				className='relative rounded-xl overflow-hidden shadow-lg'
				style={{
					aspectRatio: '2/3',
					background: gradient,
					border: '1px solid rgba(255,255,255,0.07)',
				}}
			>
				{posterUrl ? (
					<img
						src={posterUrl}
						alt={movie.title}
						className='w-full h-full object-cover transition-transform duration-300 group-hover:scale-105'
					/>
				) : (
					<div className='absolute inset-0 flex items-end p-2'>
						<span className='text-white text-xs font-semibold leading-snug drop-shadow'>
							{movie.title}
						</span>
					</div>
				)}
				<div className='absolute inset-0 bg-black opacity-0 group-hover:opacity-20 transition-opacity' />
			</div>
			<p className='mt-2 text-xs text-zinc-300 font-medium truncate leading-snug'>
				{movie.title}
			</p>
			{movie.year && <p className='text-xs text-zinc-500'>{movie.year}</p>}
			{movie.avg_rating && (
				<p className='text-xs text-yellow-400 font-semibold'>
					★ {movie.avg_rating.toFixed(1)}
				</p>
			)}
		</Link>
	)
}

// ── Sign-in gate ─────────────────────────────────────────────────────────────
function SignInGate({ action }: { action: string }) {
	return (
		<div className='flex items-center gap-3 py-1'>
			<svg
				xmlns='http://www.w3.org/2000/svg'
				className='w-5 h-5 text-zinc-500 flex-shrink-0'
				viewBox='0 0 24 24'
				fill='none'
				stroke='currentColor'
				strokeWidth={2}
			>
				<rect x='3' y='11' width='18' height='11' rx='2' ry='2' />
				<path d='M7 11V7a5 5 0 0 1 10 0v4' />
			</svg>
			<p className='text-sm text-zinc-400'>
				<Link
					href='/login'
					className='text-white font-semibold hover:underline'
				>
					Sign in
				</Link>{' '}
				to {action}
			</p>
		</div>
	)
}

// ── Main component ────────────────────────────────────────────────────────────
interface Props {
	movie: Movie
}

export default function MoviePageInteractive({ movie }: Props) {
	const { tr } = useTranslation()
	// ── user actions
	const [watched, setWatched] = useState(false)
	const [inWatchlist, setInWatchlist] = useState(false)
	const [userRating, setUserRating] = useState(0)
	const [review, setReview] = useState('')
	const [saved, setSaved] = useState(false)
	const [isSaving, setIsSaving] = useState(false)
	const [saveError, setSaveError] = useState<string | null>(null)
	const [loggedIn, setLoggedIn] = useState(false)
	const [currentUserId, setCurrentUserId] = useState<number | null>(
		getAuthUser()?.id ?? null,
	)

	// ── public movie reviews (all users)
	const [communityReviews, setCommunityReviews] = useState<PublicMovieReview[]>([])

	// ── similar movies
	const [similar, setSimilar] = useState<SimMovie[]>([])
	const [similarModel, setSimilarModel] = useState('')
	const scrollRef = useRef<HTMLDivElement>(null)

	// ── Auth state sync
	useEffect(() => {
		setLoggedIn(checkIsLoggedIn())
		setCurrentUserId(getAuthUser()?.id ?? null)
		const sync = () => {
			setLoggedIn(checkIsLoggedIn())
			setCurrentUserId(getAuthUser()?.id ?? null)
		}
		window.addEventListener('auth-change', sync)
		return () => window.removeEventListener('auth-change', sync)
	}, [])

	// ── Init from DB
	useEffect(() => {
		const token = getToken()
		setUserRating(0)
		setReview('')
		if (!token) {
			setWatched(false)
			setInWatchlist(false)
			return
		}
		// Watched + Watchlist + own review from DB
		fetchWatched(token).then(wl => {
			setWatched(wl.some(w => w.movie_id === movie.id))
		})
		fetchWatchlist(token).then(wl => {
			setInWatchlist(wl.some(w => w.movie_id === movie.id))
		})
		fetchReviews(token).then(all => {
			const ownMovieReviews = all
				.filter(r => r.movie_id === movie.id)
				.sort(
					(a, b) =>
						new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
				)

			const mine = ownMovieReviews[0]
			if (mine) {
				setUserRating(mine.rating)
				setReview(mine.review_text ?? '')
			} else {
				setUserRating(0)
				setReview('')
			}
		})
	}, [movie.id])

	useEffect(() => {
		fetchMovieReviews(movie.id, 30).then(setCommunityReviews)
	}, [movie.id])

	// ── Load similar movies (ALS cosine similarity via API)
	useEffect(() => {
		fetchSimilarMovies(movie.id, 24)
			.then(({ movies, model }) => {
				setSimilar(movies)
				setSimilarModel(model)
			})
			.catch(() => { })
	}, [movie.id])

	// ── Save review
	const handleSave = async () => {
		const token = getToken()
		if (!token) return
		if (userRating < 1 || userRating > 5) {
			setSaveError(tr.movie.ratingRequired)
			return
		}

		setIsSaving(true)
		setSaveError(null)
		const savedReview = await upsertReview(token, movie.id, userRating, review)
		if (!savedReview) {
			setIsSaving(false)
			setSaveError(tr.movie.saveError)
			return
		}

		setUserRating(savedReview.rating)
		setReview(savedReview.review_text ?? '')
		await trackKpi('rating_submit', 'movie_page', movie.id)
		if ((savedReview.review_text ?? '').trim()) {
			await trackKpi('review_submit', 'movie_page', movie.id)
		}
		setWatched(true)
		setInWatchlist(false)
		fetchMovieReviews(movie.id, 30).then(setCommunityReviews)
		setIsSaving(false)
		setSaved(true)
		setTimeout(() => setSaved(false), 2000)
	}

	// ── Horizontal scroll helpers
	const scrollBy = (dir: -1 | 1) => {
		scrollRef.current?.scrollBy({ left: dir * 600, behavior: 'smooth' })
	}

	const sectionBox = 'rounded-2xl p-6'
	const sectionStyle = {
		background: 'rgba(255,255,255,0.03)',
		border: '1px solid rgba(255,255,255,0.07)',
	}

	return (
		<div className='flex flex-col gap-8'>
			{/* ── Your rating + Watched + Watchlist + External links ─────── */}
			<div className={sectionBox} style={sectionStyle}>
				<h2 className='text-xs uppercase tracking-widest text-zinc-500 mb-4'>
					{tr.movie.yourRating}
				</h2>
				<div className='flex flex-wrap items-center justify-between gap-4'>
					{loggedIn ? (
						<div className='flex flex-wrap items-center gap-4'>
							<StarRating
								value={userRating}
								onChange={setUserRating}
								size='lg'
							/>

							{/* Watched */}
							<button
								onClick={async () => {
									const token = getToken()
									if (!token) return
									if (watched) {
										await removeWatchedDB(token, movie.id)
										setWatched(false)
									} else {
										await addWatchedDB(token, movie.id)
										setWatched(true)
									}
								}}
								className='flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all'
								style={{
									background: watched
										? 'rgba(46,160,67,0.18)'
										: 'rgba(255,255,255,0.06)',
									border: `1px solid ${watched ? 'rgba(46,160,67,0.5)' : 'rgba(255,255,255,0.12)'}`,
									color: watched ? '#4ade80' : '#a1a1aa',
								}}
							>
								{watched ? (
									<svg
										xmlns='http://www.w3.org/2000/svg'
										viewBox='0 0 20 20'
										fill='currentColor'
										className='w-4 h-4'
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
										className='w-4 h-4'
									>
										<path d='M10.75 4.75a.75.75 0 0 0-1.5 0v4.5h-4.5a.75.75 0 0 0 0 1.5h4.5v4.5a.75.75 0 0 0 1.5 0v-4.5h4.5a.75.75 0 0 0 0-1.5h-4.5v-4.5Z' />
									</svg>
								)}
								{watched ? tr.movie.watched : tr.movie.markWatched}
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
										await apiAddWatchlist(token, movie.id)
										setInWatchlist(true)
									}
								}}
								className='flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all'
								style={{
									background: inWatchlist
										? 'rgba(229,9,20,0.13)'
										: 'rgba(255,255,255,0.06)',
									border: `1px solid ${inWatchlist ? 'rgba(229,9,20,0.4)' : 'rgba(255,255,255,0.12)'}`,
									color: inWatchlist ? '#e50914' : '#a1a1aa',
								}}
							>
								<svg
									xmlns='http://www.w3.org/2000/svg'
									viewBox='0 0 20 20'
									fill='currentColor'
									className='w-4 h-4'
								>
									<path d='M6.3 2.84A1.5 1.5 0 0 0 5 4.312v11.376a.5.5 0 0 0 .77.419l4.23-2.791 4.23 2.79a.5.5 0 0 0 .77-.418V4.313a1.5 1.5 0 0 0-1.3-1.472A42.5 42.5 0 0 0 10 2.5a42.5 42.5 0 0 0-3.7.34Z' />
								</svg>
								{inWatchlist ? tr.movie.inWatchlist : tr.movie.addWatchlist}
							</button>
						</div>
					) : (
						<SignInGate action={tr.movie.signInAction} />
					)}

					{/* Right: IMDB + TMDB links */}
					<div className='flex items-center gap-2 ml-auto'>
						{movie.imdb_id && (
							<a
								href={`https://www.imdb.com/title/tt${String(movie.imdb_id).padStart(7, '0')}/`}
								target='_blank'
								rel='noopener noreferrer'
								className='flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-bold transition-opacity hover:opacity-80'
								style={{ background: '#f5c518', color: '#000' }}
							>
								{tr.movie.imdb}
							</a>
						)}
						{movie.tmdb_id && (
							<a
								href={`https://www.themoviedb.org/movie/${movie.tmdb_id}`}
								target='_blank'
								rel='noopener noreferrer'
								className='flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-bold text-white transition-opacity hover:opacity-80'
								style={{ background: '#01b4e4' }}
							>
								{tr.movie.tmdb}
							</a>
						)}
					</div>
				</div>
			</div>

			{/* ── Write review ─────────────────────────────────────────────── */}
			<div className={sectionBox} style={sectionStyle}>
				<h2 className='text-xs uppercase tracking-widest text-zinc-500 mb-4'>
					{tr.movie.yourReview}
				</h2>
				{loggedIn ? (
					<>
						<textarea
							value={review}
							onChange={e => setReview(e.target.value)}
							placeholder={tr.movie.writeReview}
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
								disabled={isSaving || userRating === 0}
								className='px-6 py-2 rounded-full font-semibold text-sm text-white transition-opacity disabled:opacity-40 disabled:cursor-not-allowed'
								style={{ background: 'var(--netflix-red)' }}
							>
								{isSaving ? tr.movie.saving : tr.movie.save}
							</button>
							{saved && (
								<span className='text-sm text-green-400 animate-pulse'>
									Saved!
								</span>
							)}
							{saveError && (
								<span className='text-sm text-red-400'>{saveError}</span>
							)}
						</div>
					</>
				) : (
					<SignInGate action={tr.movie.signInAction} />
				)}
			</div>

			{/* ── Community reviews ────────────────────────────────────────── */}
			<div>
				<h2 className='text-lg font-bold text-white mb-4'>
					{tr.movie.communityReviews}
					{communityReviews.length > 0 && (
						<span className='ml-2 text-sm font-normal text-zinc-500'>
							({communityReviews.length})
						</span>
					)}
				</h2>
				{communityReviews.length === 0 ? (
					<div
						className='rounded-2xl px-6 py-8 text-center text-zinc-500 text-sm'
						style={{
							background: 'rgba(255,255,255,0.02)',
							border: '1px solid rgba(255,255,255,0.06)',
						}}
					>
						No reviews yet — be the first to rate this movie.
					</div>
				) : (
					<div className='flex flex-col gap-4'>
						{communityReviews.map(r => (
							<div
								key={r.id}
								className='rounded-2xl px-6 py-4 flex flex-col gap-2'
								style={{
									background: 'rgba(255,255,255,0.03)',
									border: '1px solid rgba(255,255,255,0.07)',
								}}
							>
								<div className='flex items-center justify-between gap-4'>
									<div className='flex items-center gap-3'>
										<div
											className='w-8 h-8 rounded-full flex items-center justify-center overflow-hidden'
											style={{ background: 'rgba(255,255,255,0.06)' }}
										>
											<AvatarIcon avatarId={r.user_avatar_id} size={24} />
										</div>
										<div className='flex flex-col gap-0.5'>
											<Link
												href={
													r.user_id === currentUserId
														? '/profile'
														: `/profile/${r.user_id}`
												}
												className='text-sm text-zinc-200 font-medium hover:underline'
											>
												{r.user_login}
											</Link>
											<StarRating value={r.rating} size='sm' />
										</div>
									</div>
									<span className='text-xs text-zinc-500'>
										{new Date(r.created_at).toLocaleDateString('en-US', {
											month: 'short',
											day: 'numeric',
											year: 'numeric',
										})}
									</span>
								</div>
								{r.review_text?.trim() && (
									<p className='text-sm text-zinc-300 leading-relaxed pl-11'>
										{r.review_text}
									</p>
								)}
							</div>
						))}
					</div>
				)}
			</div>

			{/* ── Similar movies ───────────────────────────────────────────── */}
			{similar.length > 0 && (
				<div>
					<div className='flex items-baseline mb-4'>
						<h2 className='text-lg font-bold text-white'>{tr.movie.similar}</h2>
						{similarModel && (
							<span className='ml-auto text-xs text-zinc-500'>
								{similarModel === 'als_cosine'
									? 'iALS · Cosine Similarity'
									: similarModel === 'genre_jaccard'
										? 'Genre Jaccard fallback'
										: null}
							</span>
						)}
					</div>
					<div className='relative'>
						{/* Left arrow */}
						<button
							onClick={() => scrollBy(-1)}
							className='absolute left-0 top-1/2 -translate-y-1/2 z-10 w-9 h-9 flex items-center justify-center rounded-full text-white -ml-4'
							style={{
								background: 'rgba(0,0,0,0.7)',
								border: '1px solid rgba(255,255,255,0.12)',
							}}
						>
							<svg
								xmlns='http://www.w3.org/2000/svg'
								className='w-4 h-4'
								fill='none'
								viewBox='0 0 24 24'
								stroke='currentColor'
								strokeWidth={2.5}
							>
								<path
									strokeLinecap='round'
									strokeLinejoin='round'
									d='M15 19l-7-7 7-7'
								/>
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
							style={{
								background: 'rgba(0,0,0,0.7)',
								border: '1px solid rgba(255,255,255,0.12)',
							}}
						>
							<svg
								xmlns='http://www.w3.org/2000/svg'
								className='w-4 h-4'
								fill='none'
								viewBox='0 0 24 24'
								stroke='currentColor'
								strokeWidth={2.5}
							>
								<path
									strokeLinecap='round'
									strokeLinejoin='round'
									d='M9 5l7 7-7 7'
								/>
							</svg>
						</button>
					</div>
				</div>
			)}
		</div>
	)
}
