# Frontend

Next.js 14 приложение — главная страница с топ популярных фильмов.

## Запуск

```bash
cd frontend
npm install
npm run dev      # http://localhost:3000

# или через make (если npm в PATH)
make frontend
```

## Структура

```
frontend/
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
├── next.config.mjs
├── Dockerfile.dev          ← для docker compose
└── src/
    ├── app/
    │   ├── layout.tsx       ← корневой layout (хедер + body)
    │   ├── page.tsx         ← главная страница /
    │   └── globals.css      ← Tailwind base + кастомные стили
    ├── components/
    │   └── MovieCard.tsx    ← карточка одного фильма
    └── lib/
        └── api.ts           ← HTTP клиент, TypeScript типы
```

## Зависимости

```json
{
  "next": "14.2.3",           // App Router, Server Components, ISR
  "react": "^18",
  "tailwindcss": "^3.4.3",   // utility-first CSS
  "typescript": "^5"
}
```

---

## `src/lib/api.ts` — HTTP клиент

Единственный файл для обращений к бэкенду. Все типы совпадают с Pydantic схемами.

### Типы

```typescript
interface Movie {
    id: number;
    title: string;
    genres: string | null;      // "Action|Drama" или null
    year: number | null;
    avg_rating: number | null;
    num_ratings: number | null;
    popularity_score: number | null;
}

interface PopularMoviesResponse {
    total_returned: number;
    offset: number;
    movies: Movie[];
}
```

### `fetchPopularMovies(limit, offset)`

```typescript
export async function fetchPopularMovies(
    limit = 20,
    offset = 0
): Promise<PopularMoviesResponse>
```

- Использует нативный `fetch` с `next: { revalidate: 3600 }` — Next.js ISR кеширует ответ на 1 час.
- `API_URL` берёт из `NEXT_PUBLIC_API_URL` (env), fallback — `http://localhost:8000`.
- Бросает `Error` при не-2xx ответе.

### Конфигурация через `.env.local`

```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## `src/app/layout.tsx` — Root Layout

Обёртка для всех страниц: хедер с логотипом, `<main>` с `max-w-7xl`.
Подключает `globals.css` (Tailwind).

```tsx
export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <body>
                <header>🎬 RecSys</header>
                <main>{children}</main>
            </body>
        </html>
    );
}
```

---

## `src/app/page.tsx` — Главная страница

**Server Component** — данные фетчатся на сервере, HTML отдаётся браузеру готовым.

```tsx
export default async function HomePage() {
    const data = await fetchPopularMovies(40);  // 40 фильмов
    // ...рендер сетки карточек
}
```

- Грид: 1 колонка (mobile) → 2 → 3 → 4 (desktop `lg:`)
- Заголовок + подпись с формулой popularity

---

## `src/components/MovieCard.tsx` — Карточка фильма

Принимает `movie: Movie` и `rank: number`.

**Что отображает:**

- Порядковый номер (крупный серый текст)
- Название (`line-clamp-2` — максимум 2 строки)
- Год выхода (если есть)
- До 3 жанровых бейджей
- Средний рейтинг ★ жёлтым
- Количество оценок в тысячах (например `49k ratings`)

**Стили:** `bg-zinc-900` карточка, `hover:bg-zinc-800` при наведении, `rounded-xl`.

---

## Переменные окружения

| Переменная | Где | Описание |
|------------|-----|----------|
| `NEXT_PUBLIC_API_URL` | `.env.local` | URL бэкенда, доступен в браузере |

`NEXT_PUBLIC_` префикс обязателен для переменных, которые читаются на клиенте.

---

## Добавление новой страницы — пример

Страница поиска `/search?q=matrix`:

```
frontend/src/app/search/
└── page.tsx
```

```tsx
// src/app/search/page.tsx
import { fetchPopularMovies } from "@/lib/api";
import { MovieCard } from "@/components/MovieCard";

interface Props {
    searchParams: { q?: string };
}

export default async function SearchPage({ searchParams }: Props) {
    const q = searchParams.q ?? "";
    const data = await fetchPopularMovies(100);

    const results = data.movies.filter(m =>
        m.title.toLowerCase().includes(q.toLowerCase())
    );

    return (
        <div>
            <h1>Поиск: «{q}»</h1>
            <p>{results.length} результатов</p>
            <div className="grid grid-cols-4 gap-4">
                {results.map((m, i) => <MovieCard key={m.id} movie={m} rank={i + 1} />)}
            </div>
        </div>
    );
}
```
