'use client'

import type {
	AdminOverview,
	AdminUser,
	DailyActivity,
	RatingDist,
	TopMovie,
} from '@/lib/api'
import {
	fetchAdminKpi,
	fetchAdminDaily,
	fetchAdminOverview,
	fetchAdminRatingDist,
	fetchAdminTopMovies,
	fetchAdminUsers,
} from '@/lib/api'
import { getAuthUser, getToken, isLoggedIn } from '@/lib/authStore'
import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'

// ── Tiny SVG line chart ───────────────────────────────────────────────────────
function LineChart({
	data,
	keys,
	colors,
	labels,
}: {
	data: DailyActivity[]
	keys: (keyof DailyActivity)[]
	colors: string[]
	labels: string[]
}) {
	if (!data.length)
		return (
			<div className='h-32 flex items-center justify-center text-zinc-600 text-sm'>
				No data
			</div>
		)

	const W = 560
	const H = 130
	const pad = { t: 8, r: 8, b: 24, l: 28 }
	const iw = W - pad.l - pad.r
	const ih = H - pad.t - pad.b
	const maxVal = Math.max(1, ...data.flatMap(d => keys.map(k => Number(d[k]))))
	const xs = data.map((_, i) => pad.l + (i / Math.max(data.length - 1, 1)) * iw)
	const ys = (key: keyof DailyActivity) =>
		data.map(d => pad.t + ih - (Number(d[key]) / maxVal) * ih)
	const path = (yArr: number[]) =>
		yArr
			.map(
				(y, i) => `${i === 0 ? 'M' : 'L'}${xs[i].toFixed(1)},${y.toFixed(1)}`,
			)
			.join(' ')
	const every = Math.max(1, Math.ceil(data.length / 7))

	return (
		<svg viewBox={`0 0 ${W} ${H}`} className='w-full' style={{ height: H }}>
			{[0, 0.5, 1].map(t => (
				<line
					key={t}
					x1={pad.l}
					x2={W - pad.r}
					y1={pad.t + ih * (1 - t)}
					y2={pad.t + ih * (1 - t)}
					stroke='rgba(255,255,255,0.06)'
					strokeWidth={1}
				/>
			))}
			{[0, 0.5, 1].map(t => (
				<text
					key={t}
					x={pad.l - 4}
					y={pad.t + ih * (1 - t) + 4}
					textAnchor='end'
					fontSize={9}
					fill='rgba(255,255,255,0.3)'
				>
					{Math.round(maxVal * t)}
				</text>
			))}
			{data.map(
				(d, i) =>
					i % every === 0 && (
						<text
							key={i}
							x={xs[i]}
							y={H - 4}
							textAnchor='middle'
							fontSize={9}
							fill='rgba(255,255,255,0.3)'
						>
							{d.date.slice(5)}
						</text>
					),
			)}
			{keys.map((k, ki) => (
				<path
					key={String(k)}
					d={path(ys(k))}
					fill='none'
					stroke={colors[ki]}
					strokeWidth={2}
					strokeLinejoin='round'
					strokeLinecap='round'
				/>
			))}
			{keys.map((k, ki) => {
				const last = data.length - 1
				return (
					<circle
						key={String(k)}
						cx={xs[last]}
						cy={ys(k)[last]}
						r={3}
						fill={colors[ki]}
					/>
				)
			})}
		</svg>
	)
}

// ── Bar chart for rating distribution ────────────────────────────────────────
function RatingBar({ data }: { data: RatingDist[] }) {
	const max = Math.max(1, ...data.map(d => d.count))
	const STARS = ['★', '★★', '★★★', '★★★★', '★★★★★']
	const COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#10b981']
	return (
		<div className='space-y-2'>
			{data.map((d, i) => (
				<div key={d.rating} className='flex items-center gap-3'>
					<span className='text-xs w-16 text-zinc-500 flex-shrink-0'>
						{STARS[i]}
					</span>
					<div className='flex-1 h-5 rounded-md overflow-hidden bg-zinc-800'>
						<div
							className='h-full rounded-md transition-all duration-700'
							style={{
								width: `${(d.count / max) * 100}%`,
								background: COLORS[i],
							}}
						/>
					</div>
					<span className='text-xs text-zinc-400 w-8 text-right flex-shrink-0'>
						{d.count}
					</span>
				</div>
			))}
		</div>
	)
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({
	label,
	value,
	sub,
	accent,
}: {
	label: string
	value: string | number
	sub?: string
	accent?: string
}) {
	return (
		<div
			className='rounded-2xl p-5 flex flex-col gap-1'
			style={{
				background: 'var(--bg-card)',
				border: '1px solid rgba(255,255,255,0.07)',
			}}
		>
			<p className='text-xs text-zinc-500 uppercase tracking-wider'>{label}</p>
			<p className='text-3xl font-black' style={{ color: accent ?? 'white' }}>
				{value}
			</p>
			{sub && <p className='text-xs text-zinc-600'>{sub}</p>}
		</div>
	)
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function AdminPage() {
	const router = useRouter()
	const [allowed, setAllowed] = useState(false)
	const [token, setToken] = useState<string | null>(null)

	// Data states
	const [overview, setOverview] = useState<AdminOverview | null>(null)
	const [kpi, setKpi] = useState<{
		rec_ctr_percent: number
		feedback_session_share_percent: number
		rec_impressions: number
		sessions_started: number
	} | null>(null)
	const [daily, setDaily] = useState<DailyActivity[]>([])
	const [topMovies, setTopMovies] = useState<{
		most_watched: TopMovie[]
		top_rated: TopMovie[]
	} | null>(null)
	const [ratingDist, setRatingDist] = useState<RatingDist[]>([])
	const [users, setUsers] = useState<AdminUser[]>([])
	const [usersTotal, setUsersTotal] = useState(0)
	const [chartDays, setChartDays] = useState(30)
	const [chartKey, setChartKey] = useState<
		'new_watched' | 'new_reviews' | 'new_users' | 'new_watchlist'
	>('new_watched')
	const [loading, setLoading] = useState(true)

	// Auth check
	useEffect(() => {
		if (!isLoggedIn()) {
			router.replace('/login')
			return
		}
		const user = getAuthUser()
		if (user?.role !== 'admin') {
			router.replace('/')
			return
		}
		const t = getToken()
		setToken(t)
		setAllowed(true)
	}, [router])

	// Fetch all data once allowed
	useEffect(() => {
		if (!token) return
		setLoading(true)
		Promise.all([
			fetchAdminOverview(token),
			fetchAdminKpi(token, 7),
			fetchAdminDaily(token, chartDays),
			fetchAdminTopMovies(token),
			fetchAdminRatingDist(token),
			fetchAdminUsers(token, 50, 0),
		])
			.then(([ov, kp, dl, tm, rd, us]) => {
				setOverview(ov)
				setKpi(kp)
				setDaily(dl)
				setTopMovies(tm)
				setRatingDist(rd)
				setUsers(us.users)
				setUsersTotal(us.total)
			})
			.finally(() => setLoading(false))
	}, [token, chartDays])

	if (!allowed)
		return (
			<div
				className='min-h-screen flex items-center justify-center'
				style={{ background: 'var(--bg-primary)' }}
			>
				<div className='w-8 h-8 rounded-full border-2 border-red-600 border-t-transparent animate-spin' />
			</div>
		)

	const CHART_OPTIONS = [
		{ key: 'new_watched' as const, label: 'Watched', color: '#e50914' },
		{ key: 'new_reviews' as const, label: 'Reviews', color: '#f59e0b' },
		{ key: 'new_users' as const, label: 'Users', color: '#3b82f6' },
		{ key: 'new_watchlist' as const, label: 'Watchlist', color: '#10b981' },
	]
	const activeChart = CHART_OPTIONS.find(o => o.key === chartKey)!

	return (
		<div
			className='min-h-screen pt-24 pb-20 px-4 md:px-8'
			style={{ background: 'var(--bg-primary)' }}
		>
			<div className='max-w-6xl mx-auto space-y-8'>
				{/* ── Page header ── */}
				<div className='flex items-center gap-4'>
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
								d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'
							/>
						</svg>
					</div>
					<div>
						<h1 className='text-2xl font-black text-white'>Admin Dashboard</h1>
						<p className='text-sm text-zinc-500'>
							User activity & platform metrics
						</p>
					</div>
				</div>

				{loading ? (
					<div className='flex items-center justify-center py-24'>
						<div className='w-8 h-8 rounded-full border-2 border-red-600 border-t-transparent animate-spin' />
					</div>
				) : (
					<>
						{/* ── Stat cards ── */}
						<div className='grid grid-cols-2 md:grid-cols-4 gap-4'>
							<StatCard
								label='Total Users'
								value={overview?.total_users ?? 0}
								sub={`+${overview?.users_today ?? 0} today`}
								accent='#e50914'
							/>
							<StatCard
								label='Movies Watched'
								value={overview?.total_watched ?? 0}
							/>
							<StatCard
								label='Reviews'
								value={overview?.total_reviews ?? 0}
								sub={
									overview?.avg_rating
										? `avg ★ ${overview.avg_rating}`
										: undefined
								}
								accent='#f59e0b'
							/>
							<StatCard
								label='Watchlist Adds'
								value={overview?.total_watchlist ?? 0}
								sub={`${overview?.active_today ?? 0} active today`}
								accent='#10b981'
							/>
						</div>

						<div className='grid grid-cols-1 md:grid-cols-2 gap-4'>
							<StatCard
								label='KPI: Rec CTR (7d)'
								value={`${(kpi?.rec_ctr_percent ?? 0).toFixed(2)}%`}
								sub={`${kpi?.rec_impressions ?? 0} impressions`}
								accent='#60a5fa'
							/>
							<StatCard
								label='KPI: Sessions With Feedback (7d)'
								value={`${(kpi?.feedback_session_share_percent ?? 0).toFixed(2)}%`}
								sub={`${kpi?.sessions_started ?? 0} sessions`}
								accent='#22c55e'
							/>
						</div>

						{/* ── Activity chart ── */}
						<div
							className='rounded-2xl p-6'
							style={{
								background: 'var(--bg-card)',
								border: '1px solid rgba(255,255,255,0.07)',
							}}
						>
							<div className='flex flex-wrap items-center justify-between gap-4 mb-5'>
								<div>
									<h2 className='text-base font-bold text-white'>
										Activity Over Time
									</h2>
									<p className='text-xs text-zinc-500'>Daily platform events</p>
								</div>
								<div className='flex items-center gap-3'>
									{/* Metric selector */}
									<div className='flex gap-1'>
										{CHART_OPTIONS.map(o => (
											<button
												key={o.key}
												onClick={() => setChartKey(o.key)}
												className='px-2.5 py-1 rounded-lg text-xs font-medium transition-colors'
												style={
													chartKey === o.key
														? {
																background: o.color + '22',
																color: o.color,
																border: `1px solid ${o.color}55`,
															}
														: {
																background: 'transparent',
																color: '#71717a',
																border: '1px solid rgba(255,255,255,0.07)',
															}
												}
											>
												{o.label}
											</button>
										))}
									</div>
									{/* Day range */}
									<select
										value={chartDays}
										onChange={e => setChartDays(Number(e.target.value))}
										className='bg-zinc-800 text-white text-xs rounded-lg px-2 py-1 border border-zinc-700 outline-none'
									>
										<option value={7}>7d</option>
										<option value={14}>14d</option>
										<option value={30}>30d</option>
										<option value={60}>60d</option>
										<option value={90}>90d</option>
									</select>
								</div>
							</div>
							<LineChart
								data={daily}
								keys={[chartKey]}
								colors={[activeChart.color]}
								labels={[activeChart.label]}
							/>
						</div>

						{/* ── Rating distribution + Top movies ── */}
						<div className='grid md:grid-cols-2 gap-6'>
							{/* Rating bar chart */}
							<div
								className='rounded-2xl p-6'
								style={{
									background: 'var(--bg-card)',
									border: '1px solid rgba(255,255,255,0.07)',
								}}
							>
								<h2 className='text-base font-bold text-white mb-1'>
									Rating Distribution
								</h2>
								<p className='text-xs text-zinc-500 mb-5'>
									Stars breakdown across all reviews
								</p>
								<RatingBar data={ratingDist} />
							</div>

							{/* Top rated */}
							<div
								className='rounded-2xl p-6'
								style={{
									background: 'var(--bg-card)',
									border: '1px solid rgba(255,255,255,0.07)',
								}}
							>
								<h2 className='text-base font-bold text-white mb-1'>
									Top Rated
								</h2>
								<p className='text-xs text-zinc-500 mb-5'>
									Highest avg rating with at least 1 review
								</p>
								<div className='space-y-2'>
									{topMovies?.top_rated.slice(0, 6).map((m, i) => (
										<div key={m.movie_id} className='flex items-center gap-3'>
											<span className='text-xs text-zinc-600 w-4 text-right flex-shrink-0'>
												{i + 1}
											</span>
											<p className='flex-1 text-sm text-white truncate'>
												{m.title}
											</p>
											<span className='text-xs text-yellow-400 flex-shrink-0'>
												★ {m.avg_rating}
											</span>
											<span className='text-xs text-zinc-600 flex-shrink-0'>
												({m.review_count})
											</span>
										</div>
									))}
									{!topMovies?.top_rated.length && (
										<p className='text-sm text-zinc-600'>No reviews yet</p>
									)}
								</div>
							</div>
						</div>

						{/* ── Most watched ── */}
						<div
							className='rounded-2xl p-6'
							style={{
								background: 'var(--bg-card)',
								border: '1px solid rgba(255,255,255,0.07)',
							}}
						>
							<h2 className='text-base font-bold text-white mb-1'>
								Most Watched
							</h2>
							<p className='text-xs text-zinc-500 mb-5'>
								Movies with the most watch events
							</p>
							<div className='grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2'>
								{topMovies?.most_watched.map((m, i) => {
									const max = topMovies.most_watched[0]?.watch_count ?? 1
									return (
										<div key={m.movie_id} className='flex items-center gap-3'>
											<span className='text-xs text-zinc-600 w-4 text-right flex-shrink-0'>
												{i + 1}
											</span>
											<div className='flex-1 min-w-0'>
												<p className='text-sm text-white truncate'>{m.title}</p>
												<div className='h-1.5 rounded-full bg-zinc-800 mt-1 overflow-hidden'>
													<div
														className='h-full rounded-full bg-red-600'
														style={{
															width: `${((m.watch_count ?? 0) / max) * 100}%`,
														}}
													/>
												</div>
											</div>
											<span className='text-xs text-zinc-400 flex-shrink-0'>
												{m.watch_count}
											</span>
										</div>
									)
								})}
								{!topMovies?.most_watched.length && (
									<p className='text-sm text-zinc-600'>No watch data yet</p>
								)}
							</div>
						</div>

						{/* ── Users table ── */}
						<div
							className='rounded-2xl overflow-hidden'
							style={{
								background: 'var(--bg-card)',
								border: '1px solid rgba(255,255,255,0.07)',
							}}
						>
							<div
								className='px-6 py-5 flex items-center justify-between'
								style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}
							>
								<div>
									<h2 className='text-base font-bold text-white'>Users</h2>
									<p className='text-xs text-zinc-500'>
										{usersTotal} total registered
									</p>
								</div>
							</div>
							<div className='overflow-x-auto'>
								<table className='w-full text-sm'>
									<thead>
										<tr
											style={{
												borderBottom: '1px solid rgba(255,255,255,0.06)',
											}}
										>
											{[
												'Login',
												'Email',
												'Role',
												'Joined',
												'Watched',
												'Reviews',
												'Watchlist',
											].map(h => (
												<th
													key={h}
													className='text-left px-5 py-3 text-xs text-zinc-500 font-medium uppercase tracking-wider'
												>
													{h}
												</th>
											))}
										</tr>
									</thead>
									<tbody>
										{users.map((u, i) => (
											<tr
												key={u.id}
												style={{
													borderBottom:
														i < users.length - 1
															? '1px solid rgba(255,255,255,0.04)'
															: 'none',
												}}
												className='hover:bg-white/[0.03] transition-colors'
											>
												<td className='px-5 py-3 text-white font-medium'>
													{u.login}
												</td>
												<td className='px-5 py-3 text-zinc-400'>{u.email}</td>
												<td className='px-5 py-3'>
													<span
														className='text-xs px-2 py-0.5 rounded-full font-medium'
														style={
															u.role === 'admin'
																? {
																		background: 'rgba(229,9,20,0.2)',
																		color: '#e50914',
																	}
																: {
																		background: 'rgba(255,255,255,0.07)',
																		color: '#a1a1aa',
																	}
														}
													>
														{u.role}
													</span>
												</td>
												<td className='px-5 py-3 text-zinc-500 text-xs'>
													{u.created_at.slice(0, 10)}
												</td>
												<td className='px-5 py-3 text-zinc-300'>
													{u.watched_count}
												</td>
												<td className='px-5 py-3 text-zinc-300'>
													{u.review_count}
												</td>
												<td className='px-5 py-3 text-zinc-300'>
													{u.watchlist_count}
												</td>
											</tr>
										))}
										{!users.length && (
											<tr>
												<td
													colSpan={7}
													className='px-5 py-10 text-center text-zinc-600'
												>
													No users yet
												</td>
											</tr>
										)}
									</tbody>
								</table>
							</div>
						</div>
					</>
				)}
			</div>
		</div>
	)
}
