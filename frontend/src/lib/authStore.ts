/**
 * Client-side auth state stored in localStorage.
 * Token: "auth_token"
 * User:  "auth_user" (JSON-serialised AuthUser)
 */

import type { AuthUser } from './api'

const TOKEN_KEY = 'auth_token'
const USER_KEY = 'auth_user'

export function getToken(): string | null {
	if (typeof window === 'undefined') return null
	return localStorage.getItem(TOKEN_KEY)
}

export function getAuthUser(): AuthUser | null {
	if (typeof window === 'undefined') return null
	try {
		const raw = localStorage.getItem(USER_KEY)
		return raw ? (JSON.parse(raw) as AuthUser) : null
	} catch {
		return null
	}
}

export function setAuth(token: string, user: AuthUser): void {
	localStorage.setItem(TOKEN_KEY, token)
	localStorage.setItem(USER_KEY, JSON.stringify(user))
	// Notify other components (e.g. header) of the change
	window.dispatchEvent(new Event('auth-change'))
}

export function clearAuth(): void {
	localStorage.removeItem(TOKEN_KEY)
	localStorage.removeItem(USER_KEY)
	window.dispatchEvent(new Event('auth-change'))
}

export function isLoggedIn(): boolean {
	return !!getToken()
}
