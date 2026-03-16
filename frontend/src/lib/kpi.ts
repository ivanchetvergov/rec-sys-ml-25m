import { trackKpiEvent } from './api'
import { getToken } from './authStore'

const SESSION_KEY = 'kpi_session_id'
const SESSION_STARTED_KEY = 'kpi_session_started'

function generateSessionId(): string {
	return `sess_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`
}

export function getKpiSessionId(): string {
	if (typeof window === 'undefined') return 'server_session'
	let sid = sessionStorage.getItem(SESSION_KEY)
	if (!sid) {
		sid = generateSessionId()
		sessionStorage.setItem(SESSION_KEY, sid)
	}
	return sid
}

export async function ensureKpiSessionStarted(): Promise<void> {
	if (typeof window === 'undefined') return
	if (sessionStorage.getItem(SESSION_STARTED_KEY) === '1') return
	sessionStorage.setItem(SESSION_STARTED_KEY, '1')
	await trackKpi('session_start')
}

export async function trackKpi(
	eventType: string,
	block?: string,
	movieId?: number,
): Promise<void> {
	if (typeof window === 'undefined') return
	const token = getToken() ?? undefined
	await trackKpiEvent(getKpiSessionId(), eventType, block, movieId, token)
}
