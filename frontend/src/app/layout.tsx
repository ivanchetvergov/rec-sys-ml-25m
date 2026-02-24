import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
	title: 'RecSys',
	description: 'Movie recommendations',
}

export default function RootLayout({
	children,
}: {
	children: React.ReactNode
}) {
	return (
		<html lang='en'>
			<body>
				<header
					className='fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 py-4'
					style={{
						background:
							'linear-gradient(to bottom, rgba(0,0,0,0.9) 0%, transparent 100%)',
					}}
				>
					<div className='flex items-center gap-8'>
						<span
							className='text-2xl font-black tracking-widest'
							style={{ color: 'var(--netflix-red)', letterSpacing: '0.15em' }}
						>
							RECSYS
						</span>
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
						<a href='/login'>
							<button
								className='px-4 py-1.5 rounded-md text-sm font-bold text-white transition-all hover:opacity-80'
								style={{ background: 'var(--netflix-red)' }}
							>
								Sign In
							</button>
						</a>
					</div>
				</header>
				<main>{children}</main>
			</body>
		</html>
	)
}
