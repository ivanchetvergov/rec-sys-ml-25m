import Header from '@/components/Header'
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
				<Header />
				<main>{children}</main>
			</body>
		</html>
	)
}
