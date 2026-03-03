'use client'

import { getAuthUser, isLoggedIn } from '@/lib/authStore'
import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'

export default function AdminPage() {
	const router = useRouter()
	const [allowed, setAllowed] = useState(false)

	useEffect(() => {
		if (!isLoggedIn()) {
			router.replace('/login')
			return
		}
		const user = getAuthUser()
		if (user?.role !== 'admin') {
			console.log(user?.role, user?.login)
			router.replace('/')
			return
		}
		setAllowed(true)
	}, [router])

	if (!allowed) {
		return (
			<div
				className='min-h-screen flex items-center justify-center'
				style={{ background: 'var(--bg-primary)' }}
			>
				<div className='w-8 h-8 rounded-full border-2 border-red-600 border-t-transparent animate-spin' />
			</div>
		)
	}

	return (
		<div
			className='min-h-screen pt-24 pb-20 px-4 md:px-8'
			style={{ background: 'var(--bg-primary)' }}
		>
			<div className='relative max-w-6xl mx-auto'>
				{/* Header */}
				<div className='flex items-center gap-4 mb-8'>
					<div
						className='w-12 h-12 rounded-xl flex items-center justify-center'
						style={{
							background: 'rgba(229,9,20,0.15)',
							border: '1px solid rgba(229,9,20,0.3)',
						}}
					>
						<svg
							xmlns='http://www.w3.org/2000/svg'
							className='w-6 h-6'
							fill='none'
							viewBox='0 0 24 24'
							stroke='currentColor'
							strokeWidth={1.8}
							style={{ color: 'var(--netflix-red)' }}
						>
							<path
								strokeLinecap='round'
								strokeLinejoin='round'
								d='M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 0 0 2.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 0 0 1.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 0 0-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 0 0-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 0 0-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 0 0-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 0 0 1.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.573-1.066z'
							/>
							<circle cx='12' cy='12' r='3' />
						</svg>
					</div>
					<div>
						<h1 className='text-2xl font-black text-white'>Admin Panel</h1>
						<p className='text-sm text-zinc-500'>
							Manage users, content, and system settings
						</p>
					</div>
				</div>

				{/* Empty state */}
				<div
					className='rounded-2xl p-16 flex flex-col items-center justify-center gap-4'
					style={{
						background: 'var(--bg-card)',
						border: '1px solid rgba(255,255,255,0.07)',
					}}
				>
					<svg
						xmlns='http://www.w3.org/2000/svg'
						className='w-16 h-16 text-zinc-700'
						fill='none'
						viewBox='0 0 24 24'
						stroke='currentColor'
						strokeWidth={1}
					>
						<path
							strokeLinecap='round'
							strokeLinejoin='round'
							d='M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4'
						/>
					</svg>
					<p className='text-zinc-400 text-lg font-semibold'>Coming soon</p>
					<p className='text-zinc-600 text-sm text-center max-w-md'>
						Admin dashboard features are under development. User management,
						analytics, and content moderation will appear here.
					</p>
				</div>
			</div>
		</div>
	)
}
