'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { getAuthUser, clearAuth, isLoggedIn } from '@/lib/authStore'
import type { AuthUser } from '@/lib/api'

export default function Header() {
	const router = useRouter()
	const [user, setUser] = useState<AuthUser | null>(null)

	// Read auth state on mount and whenever auth-change fires
	useEffect(() => {
		function sync() {
			setUser(isLoggedIn() ? getAuthUser() : null)
		}
		sync()
		window.addEventListener('auth-change', sync)
		return () => window.removeEventListener('auth-change', sync)
	}, [])

	function handleLogout() {
		clearAuth()
		router.push('/')
	}

	return (
		<header
			className='fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 py-4'
			style={{
				background:
					'linear-gradient(to bottom, rgba(0,0,0,0.9) 0%, transparent 100%)',
			}}
		>
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
					<div className='flex items-center gap-3'>
						{/* Avatar + username */}
						<Link
							href='/profile'
							className='flex items-center gap-2 text-sm text-zinc-200 hover:text-white transition-colors'
						>
							<div
								className='w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white'
								style={{ background: 'var(--netflix-red)' }}
							>
								{user.login.charAt(0).toUpperCase()}
							</div>
							<span className='hidden sm:block font-medium'>{user.login}</span>
						</Link>

						{/* Logout */}
						<button
							onClick={handleLogout}
							className='px-3 py-1.5 rounded-md text-xs font-semibold text-zinc-300 border border-zinc-700 hover:border-zinc-400 hover:text-white transition-all'
						>
							Sign Out
						</button>
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
