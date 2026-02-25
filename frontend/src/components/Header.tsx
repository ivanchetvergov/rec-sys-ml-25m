'use client'

import type { AuthUser } from '@/lib/api'
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

	const menuRef = useRef<HTMLDivElement>(null)

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
				{/* Search icon */}
				<button className='text-zinc-300 hover:text-white transition-colors'>
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

				{user ? (
					/* ── Avatar with dropdown ──────────────────────────────── */
					<div className='relative' ref={menuRef}>
						{/* Avatar circle — dropdown trigger */}
						<button
							onClick={() => {
								setMenuOpen(v => !v)
								setSwitchOpen(false)
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
										<p className='text-xs text-zinc-500 truncate'>{user.email}</p>
									</div>
								</div>

								{/* ── Switch account ────────────────────────────────── */}
								<div style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
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
											{accounts.filter(a => a.user.id !== user.id).length === 0 && (
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
												<span className='text-sm text-zinc-400'>Add account</span>
											</Link>
										</div>
									)}
								</div>

								{/* ── My profile ────────────────────────────────────── */}
								<Link
									href='/profile'
									onClick={() => { setMenuOpen(false); setSwitchOpen(false) }}
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

