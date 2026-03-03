'use client'

import type { AuthUser, Movie } from '@/lib/api'
import { fetchMovieDetails, searchMovies } from '@/lib/api'
import type { StoredAccount } from '@/lib/authStore'
import {
	clearAuth,
	getAccounts,
	getAuthUser,
	isLoggedIn,
	switchAccount,
} from '@/lib/authStore'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useEffect, useRef, useState } from 'react'

export default function Header() {
	const router = useRouter()
	const [user, setUser] = useState<AuthUser | null>(null)
	const [accounts, setAccounts] = useState<StoredAccount[]>([])

	// Dropdown open state
	const [menuOpen, setMenuOpen] = useState(false)
	// Whether "Switch account" sub-panel is expanded inside the dropdown
	const [switchOpen, setSwitchOpen] = useState(false)
	// Search popup
	const [searchOpen, setSearchOpen] = useState(false)
	const [searchQuery, setSearchQuery] = useState('')
	const [searchResults, setSearchResults] = useState<Movie[]>([])
	const [searchLoading, setSearchLoading] = useState(false)
	const [posters, setPosters] = useState<Record<number, string | null>>({})

	const menuRef = useRef<HTMLDivElement>(null)
	const searchRef = useRef<HTMLDivElement>(null)

	// Sync auth state on mount and on auth-change events
	useEffect(() => {
		function sync() {
			const loggedIn = isLoggedIn()
			setUser(loggedIn ? getAuthUser() : null)
			setAccounts(getAccounts())
		}
		sync()
		window.addEventListener('auth-change', sync)
		return () => window.removeEventListener('auth-change', sync)
	}, [])

	// Close dropdown when clicking outside
	useEffect(() => {
		function handleClickOutside(e: MouseEvent) {
			if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
				setMenuOpen(false)
				setSwitchOpen(false)
			}
		}
		if (menuOpen) {
			document.addEventListener('mousedown', handleClickOutside)
		}
		return () => document.removeEventListener('mousedown', handleClickOutside)
	}, [menuOpen])

	// Close search on Escape
	useEffect(() => {
		function handleKey(e: KeyboardEvent) {
			if (e.key === 'Escape') {
				setSearchOpen(false)
				setSearchQuery('')
			}
		}
		if (searchOpen) {
			document.addEventListener('keydown', handleKey)
		}
		return () => document.removeEventListener('keydown', handleKey)
	}, [searchOpen])

	// Debounced fuzzy search
	useEffect(() => {
		const trimmed = searchQuery.trim()
		if (!trimmed) {
			setSearchResults([])
			setSearchLoading(false)
			return
		}
		setSearchLoading(true)
		const timer = setTimeout(async () => {
			try {
				const data = await searchMovies(trimmed, 12)
				setSearchResults(data.movies)
			} catch {
				setSearchResults([])
			} finally {
				setSearchLoading(false)
			}
		}, 300)
		return () => clearTimeout(timer)
	}, [searchQuery])

	// Load posters for search results
	useEffect(() => {
		if (!searchResults.length) { setPosters({}); return }
		let cancelled = false
		const ids = searchResults.map(m => m.id)
		ids.forEach(id => {
			if (posters[id] !== undefined) return  // already loaded or loading
			setPosters(p => ({ ...p, [id]: null }))  // mark as loading
			fetchMovieDetails(id).then(d => {
				if (!cancelled) setPosters(p => ({ ...p, [id]: d?.poster_url ?? '' }))
			})
		})
		return () => { cancelled = true }
	// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [searchResults])

	function handleLogout() {
		clearAuth()
		setMenuOpen(false)
		setSwitchOpen(false)
		// If still logged in (auto-switched to another account), stay; otherwise go home
		if (!isLoggedIn()) router.push('/')
	}

	function handleSwitch(userId: number) {
		switchAccount(userId)
		setMenuOpen(false)
		setSwitchOpen(false)
	}

	// Avatar letter — first char of login, uppercase
	const avatarLetter = user?.login.charAt(0).toUpperCase() ?? ''

	return (
		<header
			className='fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 py-4'
			style={{
				background:
					'linear-gradient(to bottom, rgba(0,0,0,0.9) 0%, transparent 100%)',
			}}
		>
			{/* ── Left: logo + nav ─────────────────────────────────────────── */}
			<div className='flex items-center gap-8'>
				<Link href='/'>
					<span
						className='text-2xl font-black tracking-widest cursor-pointer'
						style={{ color: 'var(--netflix-red)', letterSpacing: '0.15em' }}
					>
						RECSYS
					</span>
				</Link>
				<nav className='hidden md:flex items-center gap-6 text-sm text-zinc-300'>
					<a
						href='/'
						className='hover:text-white transition-colors font-medium text-white'
					>
						Home
					</a>
					<a href='/#popular' className='hover:text-white transition-colors'>
						Trending
					</a>
					<a href='/#catalog' className='hover:text-white transition-colors'>
						Catalog
					</a>
					<a href='/profile' className='hover:text-white transition-colors'>
						My Profile
					</a>
				</nav>
			</div>

			{/* ── Right: search + avatar/sign-in ───────────────────────────── */}
			<div className='flex items-center gap-4'>
				{/* Search trigger */}
				<button
					onClick={() => {
						setSearchOpen(true)
						setMenuOpen(false)
						setSwitchOpen(false)
					}}
					className='text-zinc-300 hover:text-white transition-colors'
					aria-label='Search'
				>
					<svg
						xmlns='http://www.w3.org/2000/svg'
						className='w-5 h-5'
						fill='none'
						viewBox='0 0 24 24'
						stroke='currentColor'
						strokeWidth={2}
					>
						<path
							strokeLinecap='round'
							strokeLinejoin='round'
							d='M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z'
						/>
					</svg>
				</button>

				{/* ── Spotlight search overlay ─────────────────────────── */}
				{searchOpen && (
					<div
						className='fixed inset-0 z-[9999] flex items-start justify-center pt-[18vh]'
						onClick={() => {
							setSearchOpen(false)
							setSearchQuery('')
						}}
					>
						{/* Backdrop */}
						<div className='absolute inset-0 bg-black/60 backdrop-blur-sm' />

						{/* Spotlight card */}
						<div
							ref={searchRef}
							onClick={e => e.stopPropagation()}
							className='relative w-full max-w-[620px] rounded-2xl overflow-hidden shadow-2xl animate-in fade-in zoom-in-95 duration-150'
							style={{
								background: 'rgba(30, 30, 30, 0.92)',
								border: '1px solid rgba(255,255,255,0.12)',
								backdropFilter: 'blur(40px)',
								boxShadow:
									'0 25px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05) inset',
							}}
						>
							{/* Search input row */}
							<div className='flex items-center gap-3 px-5 py-4'>
								<svg
									xmlns='http://www.w3.org/2000/svg'
									className='w-5 h-5 text-zinc-400 flex-shrink-0'
									fill='none'
									viewBox='0 0 24 24'
									stroke='currentColor'
									strokeWidth={2}
								>
									<path
										strokeLinecap='round'
										strokeLinejoin='round'
										d='M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z'
									/>
								</svg>
								<input
									autoFocus
									type='text'
									value={searchQuery}
									onChange={e => setSearchQuery(e.target.value)}
									placeholder='Search movies…'
									className='flex-1 bg-transparent text-lg text-white placeholder-zinc-500 outline-none'
								/>
								{searchQuery ? (
									<button
										onClick={() => setSearchQuery('')}
										className='text-zinc-500 hover:text-white transition-colors'
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
												d='M6 18L18 6M6 6l12 12'
											/>
										</svg>
									</button>
								) : (
									<kbd className='text-[11px] text-zinc-500 border border-zinc-700 rounded px-1.5 py-0.5 font-mono'>
										ESC
									</kbd>
								)}
							</div>

							{/* Divider */}
							<div
								className='h-px'
								style={{ background: 'rgba(255,255,255,0.08)' }}
							/>

							{/* Results area */}
							<div className='max-h-[50vh] overflow-y-auto'>
								{searchLoading ? (
									/* Skeleton loader */
									<div className='px-4 py-2 space-y-2'>
										{[...Array(4)].map((_, i) => (
											<div
												key={i}
												className='flex items-center gap-3 animate-pulse px-1 py-1'
											>
												<div className='w-9 h-[52px] rounded-md bg-zinc-700/60 flex-shrink-0' />
												<div className='flex-1 space-y-1.5'>
													<div className='h-3 w-3/4 rounded bg-zinc-700/60' />
													<div className='h-2.5 w-1/2 rounded bg-zinc-800/60' />
												</div>
											</div>
										))}
									</div>
								) : searchResults.length > 0 ? (
									/* Movie results */
									<div className='py-1'>
										{searchResults.map(movie => (
											<button
												key={movie.id}
												onClick={() => {
													setSearchOpen(false)
													setSearchQuery('')
													router.push(`/movies/${movie.id}`)
												}}
												className='w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-white/[0.06] transition-colors cursor-pointer'
											>
												{/* Poster thumbnail */}
												<div className='w-9 h-[52px] rounded-md overflow-hidden bg-zinc-800 flex-shrink-0 relative'>
													{posters[movie.id] ? (
														<img
															src={posters[movie.id]!}
															alt=''
															className='w-full h-full object-cover'
														/>
													) : (
														<div className='w-full h-full flex items-center justify-center'>
															<svg xmlns='http://www.w3.org/2000/svg' className='w-4 h-4 text-zinc-600' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={1.5}>
																<path strokeLinecap='round' strokeLinejoin='round' d='M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z' />
															</svg>
														</div>
													)}
												</div>
												{/* Title + meta */}
												<div className='flex-1 min-w-0'>
													<p className='text-sm text-white truncate'>
														{movie.title}
													</p>
													<p className='text-xs text-zinc-500 truncate'>
														{[
															movie.year,
															movie.genres?.split('|').slice(0, 2).join(', '),
														]
															.filter(Boolean)
															.join(' · ')}
													</p>
												</div>
												{/* Rating badge */}
												{movie.avg_rating != null && (
													<span className='text-xs text-yellow-400 flex items-center gap-0.5 flex-shrink-0'>
														<svg
															xmlns='http://www.w3.org/2000/svg'
															className='w-3 h-3'
															viewBox='0 0 20 20'
															fill='currentColor'
														>
															<path d='M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z' />
														</svg>
														{movie.avg_rating.toFixed(1)}
													</span>
												)}
											</button>
										))}
									</div>
								) : searchQuery.trim() ? (
									/* No results */
									<div className='px-5 py-8 flex flex-col items-center gap-2'>
										<p className='text-sm text-zinc-500'>No movies found</p>
										<p className='text-xs text-zinc-600'>
											Try a different spelling
										</p>
									</div>
								) : (
									/* Empty state */
									<div className='px-5 py-10 flex flex-col items-center gap-3'>
										<svg
											xmlns='http://www.w3.org/2000/svg'
											className='w-12 h-12 text-zinc-700'
											fill='none'
											viewBox='0 0 24 24'
											stroke='currentColor'
											strokeWidth={1}
										>
											<path
												strokeLinecap='round'
												strokeLinejoin='round'
												d='M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z'
											/>
										</svg>
										<p className='text-sm text-zinc-400'>
											Search for movies by title
										</p>
										<p className='text-xs text-zinc-600'>Tolerant to typos</p>
									</div>
								)}
							</div>
						</div>
					</div>
				)}

				{user ? (
					/* ── Avatar with dropdown ──────────────────────────────── */
					<div className='relative' ref={menuRef}>
						{/* Avatar circle — dropdown trigger */}
						<button
							onClick={() => {
								setMenuOpen(v => !v)
								setSwitchOpen(false)
								setSearchOpen(false)
							}}
							className='w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold text-white focus:outline-none transition-opacity hover:opacity-80 select-none'
							style={{ background: 'var(--netflix-red)' }}
							aria-label='Account menu'
						>
							{avatarLetter}
						</button>

						{/* Dropdown panel */}
						{menuOpen && (
							<div
								className='absolute right-0 mt-2 w-56 rounded-xl overflow-hidden shadow-2xl'
								style={{
									background: 'rgba(18,18,18,0.97)',
									border: '1px solid rgba(255,255,255,0.1)',
									backdropFilter: 'blur(12px)',
								}}
							>
								{/* Current user info */}
								<div
									className='flex items-center gap-3 px-4 py-3'
									style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}
								>
									<div
										className='w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-xs font-bold text-white'
										style={{ background: 'var(--netflix-red)' }}
									>
										{avatarLetter}
									</div>
									<div className='min-w-0'>
										<p className='text-sm font-semibold text-white truncate'>
											{user.login}
										</p>
										<p className='text-xs text-zinc-500 truncate'>
											{user.email}
										</p>
									</div>
								</div>

								{/* ── Switch account ────────────────────────────────── */}
								<div
									style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}
								>
									<button
										onClick={() => setSwitchOpen(v => !v)}
										className='w-full flex items-center justify-between gap-2 px-4 py-3 text-sm text-zinc-200 hover:bg-white/5 transition-colors'
									>
										<div className='flex items-center gap-3'>
											{/* people icon */}
											<svg
												xmlns='http://www.w3.org/2000/svg'
												className='w-4 h-4 text-zinc-400'
												fill='none'
												viewBox='0 0 24 24'
												stroke='currentColor'
												strokeWidth={2}
											>
												<path
													strokeLinecap='round'
													strokeLinejoin='round'
													d='M17 20h5v-2a4 4 0 0 0-4-4h-1M9 20H4v-2a4 4 0 0 1 4-4h1m8-4a4 4 0 1 1-8 0 4 4 0 0 1 8 0zM6 8a4 4 0 1 1 0-8 4 4 0 0 1 0 8z'
												/>
											</svg>
											Switch account
										</div>
										{/* chevron */}
										<svg
											xmlns='http://www.w3.org/2000/svg'
											className={`w-3.5 h-3.5 text-zinc-500 transition-transform ${switchOpen ? 'rotate-180' : ''}`}
											fill='none'
											viewBox='0 0 24 24'
											stroke='currentColor'
											strokeWidth={2.5}
										>
											<path
												strokeLinecap='round'
												strokeLinejoin='round'
												d='M19 9l-7 7-7-7'
											/>
										</svg>
									</button>

									{/* Accounts sub-list */}
									{switchOpen && (
										<div
											className='pb-2'
											style={{ background: 'rgba(255,255,255,0.03)' }}
										>
											{accounts
												.filter(a => a.user.id !== user.id)
												.map(a => (
													<button
														key={a.user.id}
														onClick={() => handleSwitch(a.user.id)}
														className='w-full flex items-center gap-3 px-4 py-2.5 hover:bg-white/5 transition-colors'
													>
														<div
															className='w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center text-xs font-bold text-white'
															style={{
																background: `hsl(${(a.user.id * 63) % 360},55%,40%)`,
															}}
														>
															{a.user.login.charAt(0).toUpperCase()}
														</div>
														<span className='text-sm text-zinc-200 truncate'>
															{a.user.login}
														</span>
													</button>
												))}

											{/* No other accounts hint */}
											{accounts.filter(a => a.user.id !== user.id).length ===
												0 && (
												<p className='px-4 py-2 text-xs text-zinc-600'>
													No other accounts saved
												</p>
											)}

											{/* Add account */}
											<Link
												href='/login'
												onClick={() => {
													setMenuOpen(false)
													setSwitchOpen(false)
												}}
												className='flex items-center gap-3 px-4 py-2.5 hover:bg-white/5 transition-colors'
											>
												<div
													className='w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center'
													style={{
														background: 'rgba(255,255,255,0.08)',
														border: '1px dashed rgba(255,255,255,0.2)',
													}}
												>
													<svg
														xmlns='http://www.w3.org/2000/svg'
														className='w-3.5 h-3.5 text-zinc-400'
														fill='none'
														viewBox='0 0 24 24'
														stroke='currentColor'
														strokeWidth={2.5}
													>
														<path
															strokeLinecap='round'
															strokeLinejoin='round'
															d='M12 4v16m8-8H4'
														/>
													</svg>
												</div>
												<span className='text-sm text-zinc-400'>
													Add account
												</span>
											</Link>
										</div>
									)}
								</div>
								{/* ── Admin Panel (admin only) ──────────────────────── */}
								{user.role === 'admin' && (
									<Link
										href='/admin'
										onClick={() => {
											setMenuOpen(false)
											setSwitchOpen(false)
										}}
										className='flex items-center gap-3 px-4 py-3 text-sm text-zinc-200 hover:bg-white/5 transition-colors'
									>
										<svg
											xmlns='http://www.w3.org/2000/svg'
											className='w-4 h-4 text-zinc-400'
											fill='none'
											viewBox='0 0 24 24'
											stroke='currentColor'
											strokeWidth={2}
										>
											<path
												strokeLinecap='round'
												strokeLinejoin='round'
												d='M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 0 0 2.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 0 0 1.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 0 0-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 0 0-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 0 0-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 0 0-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 0 0 1.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.573-1.066z'
											/>
											<circle cx='12' cy='12' r='3' />
										</svg>
										Admin Panel
									</Link>
								)}

								{/* ── My profile ────────────────────────────────────── */}
								<Link
									href='/profile'
									onClick={() => {
										setMenuOpen(false)
										setSwitchOpen(false)
									}}
									className='flex items-center gap-3 px-4 py-3 text-sm text-zinc-200 hover:bg-white/5 transition-colors'
								>
									<svg
										xmlns='http://www.w3.org/2000/svg'
										className='w-4 h-4 text-zinc-400'
										fill='none'
										viewBox='0 0 24 24'
										stroke='currentColor'
										strokeWidth={2}
									>
										<path
											strokeLinecap='round'
											strokeLinejoin='round'
											d='M16 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0zM12 14a7 7 0 0 0-7 7h14a7 7 0 0 0-7-7z'
										/>
									</svg>
									My profile
								</Link>

								{/* ── Sign out ──────────────────────────────────────── */}
								<button
									onClick={handleLogout}
									className='w-full flex items-center gap-3 px-4 py-3 text-sm text-zinc-200 hover:bg-white/5 transition-colors'
									style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}
								>
									<svg
										xmlns='http://www.w3.org/2000/svg'
										className='w-4 h-4 text-zinc-400'
										fill='none'
										viewBox='0 0 24 24'
										stroke='currentColor'
										strokeWidth={2}
									>
										<path
											strokeLinecap='round'
											strokeLinejoin='round'
											d='M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h5a2 2 0 0 1 2 2v1'
										/>
									</svg>
									Sign out
								</button>
							</div>
						)}
					</div>
				) : (
					<Link href='/login'>
						<button
							className='px-4 py-1.5 rounded-md text-sm font-bold text-white transition-all hover:opacity-80'
							style={{ background: 'var(--netflix-red)' }}
						>
							Sign In
						</button>
					</Link>
				)}
			</div>
		</header>
	)
}
