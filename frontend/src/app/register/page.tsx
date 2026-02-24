'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useState } from 'react'

export default function RegisterPage() {
	const router = useRouter()
	const [username, setUsername] = useState('')
	const [email, setEmail] = useState('')
	const [password, setPassword] = useState('')
	const [confirm, setConfirm] = useState('')
	const [error, setError] = useState<{
		field?: string
		message: string
	} | null>(null)
	const [loading, setLoading] = useState(false)

	async function handleSubmit(e: React.FormEvent) {
		e.preventDefault()
		setError(null)

		if (password !== confirm) {
			setError({ field: 'confirm', message: 'Passwords do not match' })
			return
		}

		setLoading(true)
		// TODO: replace with real API call when backend auth is ready
		await new Promise(r => setTimeout(r, 500))
		setLoading(false)
		router.push('/')
	}

	const inputClass = (field: string) =>
		`rounded-lg px-4 py-3 text-sm text-white outline-none transition-all ${
			error?.field === field
				? 'border border-red-500 bg-red-950/30'
				: 'border border-zinc-700 bg-zinc-800/60 focus:border-zinc-400'
		}`

	return (
		<div
			className='min-h-screen flex items-center justify-center px-4 py-20'
			style={{ background: 'var(--bg-primary)' }}
		>
			{/* Glow */}
			<div
				className='absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] rounded-full pointer-events-none'
				style={{
					background:
						'radial-gradient(circle, rgba(229,9,20,0.07) 0%, transparent 70%)',
				}}
			/>

			<div
				className='relative w-full max-w-md rounded-2xl p-10'
				style={{
					background: 'rgba(20,20,20,0.97)',
					border: '1px solid rgba(255,255,255,0.08)',
					boxShadow: '0 25px 60px rgba(0,0,0,0.7)',
				}}
			>
				{/* Logo */}
				<div className='mb-8'>
					<Link href='/'>
						<span
							className='text-2xl font-black tracking-widest cursor-pointer'
							style={{ color: 'var(--netflix-red)', letterSpacing: '0.15em' }}
						>
							RECSYS
						</span>
					</Link>
				</div>

				<h1 className='text-3xl font-bold text-white mb-2'>Create Account</h1>
				<p className='text-zinc-400 text-sm mb-8'>
					Already have an account?{' '}
					<Link
						href='/login'
						className='text-white hover:underline font-medium'
					>
						Sign in
					</Link>
				</p>

				<form onSubmit={handleSubmit} className='flex flex-col gap-4'>
					{/* Username */}
					<div className='flex flex-col gap-1'>
						<label className='text-xs text-zinc-400 font-medium uppercase tracking-wider'>
							Username
						</label>
						<input
							type='text'
							autoComplete='username'
							value={username}
							onChange={e => setUsername(e.target.value)}
							placeholder='johndoe'
							required
							className={inputClass('username')}
						/>
						{error?.field === 'username' && (
							<span className='text-xs text-red-400'>{error.message}</span>
						)}
					</div>

					{/* Email */}
					<div className='flex flex-col gap-1'>
						<label className='text-xs text-zinc-400 font-medium uppercase tracking-wider'>
							Email
						</label>
						<input
							type='email'
							autoComplete='email'
							value={email}
							onChange={e => setEmail(e.target.value)}
							placeholder='you@example.com'
							required
							className={inputClass('email')}
						/>
						{error?.field === 'email' && (
							<span className='text-xs text-red-400'>{error.message}</span>
						)}
					</div>

					{/* Password */}
					<div className='flex flex-col gap-1'>
						<label className='text-xs text-zinc-400 font-medium uppercase tracking-wider'>
							Password
						</label>
						<input
							type='password'
							autoComplete='new-password'
							value={password}
							onChange={e => setPassword(e.target.value)}
							placeholder='min. 6 characters'
							required
							className={inputClass('password')}
						/>
						{error?.field === 'password' && (
							<span className='text-xs text-red-400'>{error.message}</span>
						)}
					</div>

					{/* Confirm */}
					<div className='flex flex-col gap-1'>
						<label className='text-xs text-zinc-400 font-medium uppercase tracking-wider'>
							Confirm Password
						</label>
						<input
							type='password'
							autoComplete='new-password'
							value={confirm}
							onChange={e => setConfirm(e.target.value)}
							placeholder='••••••••'
							required
							className={inputClass('confirm')}
						/>
						{error?.field === 'confirm' && (
							<span className='text-xs text-red-400'>{error.message}</span>
						)}
					</div>

					{/* Generic error */}
					{error && !error.field && (
						<div className='rounded-lg px-4 py-3 text-sm text-red-300 bg-red-950/40 border border-red-800'>
							{error.message}
						</div>
					)}

					{/* Submit */}
					<button
						type='submit'
						disabled={loading}
						className='mt-2 w-full rounded-lg py-3.5 text-sm font-bold text-white transition-all duration-150 active:scale-[0.98] disabled:opacity-60'
						style={{ background: loading ? '#9b0a10' : 'var(--netflix-red)' }}
					>
						{loading ? (
							<span className='flex items-center justify-center gap-2'>
								<svg
									className='w-4 h-4 animate-spin'
									viewBox='0 0 24 24'
									fill='none'
								>
									<circle
										className='opacity-25'
										cx='12'
										cy='12'
										r='10'
										stroke='currentColor'
										strokeWidth='4'
									/>
									<path
										className='opacity-75'
										fill='currentColor'
										d='M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z'
									/>
								</svg>
								Creating account…
							</span>
						) : (
							'Create Account'
						)}
					</button>
				</form>

				<p className='mt-6 text-center text-xs text-zinc-600'>
					By registering, you agree to our{' '}
					<span className='text-zinc-400 hover:underline cursor-pointer'>
						Terms of Use
					</span>{' '}
					and{' '}
					<span className='text-zinc-400 hover:underline cursor-pointer'>
						Privacy Policy
					</span>
					.
				</p>
			</div>
		</div>
	)
}
