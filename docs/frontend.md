# Frontend (Next.js)

## 1. Назначение

Frontend отвечает за:

1. Просмотр каталога и деталей фильма.
2. Персональные рекомендации.
3. Пользовательские действия: rating/review/watchlist/watched.
4. Профиль пользователя и публичные профили.
5. Админ-страницу статистики.

## 2. Стек

- Next.js App Router
- TypeScript
- Tailwind CSS
- SSR + client components

## 3. Структура

```text
frontend/src
  app/
    page.tsx
    movies/[id]/page.tsx
    movies/[id]/MoviePageInteractive.tsx
    profile/page.tsx
    profile/[userId]/page.tsx
    login/page.tsx
    register/page.tsx
    admin/page.tsx
  components/
    Header.tsx
    MovieCard.tsx
    MovieDetailModal.tsx
    HeroSection.tsx
    CatalogSection.tsx
  lib/
    api.ts
    authStore.ts
    avatars.tsx
```

## 4. API клиент

Файл: `frontend/src/lib/api.ts`

Особенности:

1. SSR использует `BACKEND_INTERNAL_URL`.
2. Browser использует `NEXT_PUBLIC_API_URL`.
3. Для 401 применяется centralized handler:
   - `clearAuth()`

Покрываемые домены API:

- movies
- auth
- watchlist
- watched
- reviews
- users profile/privacy
- admin

## 5. Auth state

Файл: `frontend/src/lib/authStore.ts`

Хранилище:

- `auth_token`
- `auth_user`
- `auth_accounts` (multi-account)

Возможности:

1. Проверка exp в JWT.
2. Автоочистка протухшей сессии.
3. Переключение аккаунтов.
4. Персистентный avatar_id на аккаунт.

## 6. Главные пользовательские flow

### 6.1 Фильм

Файл: `app/movies/[id]/MoviePageInteractive.tsx`

Потоки:

1. Загрузка собственного состояния: watched, watchlist, review.
2. Сохранение review через upsert.
3. Обновление community reviews.
4. Показ similar movies.
5. Ссылки на профиль автора review:
   - свой review -> /profile
   - чужой review -> /profile/{id}

### 6.2 Профиль

Файл: `app/profile/page.tsx`

Показывает:

1. Watched list
2. Watchlist
3. Ratings & Reviews
4. Genre stats
5. Profile privacy toggle

Навигация:

- Из карточек и review title есть переход на `/movies/{id}`.

### 6.3 Публичный профиль

Файл: `app/profile/[userId]/page.tsx`

- Загружает public profile
- Учитывает privacy блокировку через backend response

## 7. Header и UX

Файл: `components/Header.tsx`

Возможности:

1. Search overlay с fuzzy search.
2. Account dropdown.
3. Account switch.
4. Avatar rendering.

## 8. Модалка фильма

Файл: `components/MovieDetailModal.tsx`

Поведение синхронизировано с page-interactive flow:

1. prefill review/rating
2. save review
3. watched/watchlist toggle

## 9. Caching стратегия

1. personal recs: `no-store`
2. similar movies: `no-store`
3. movie details: revalidate 24h
4. popular movies: ISR revalidate

## 10. Практические рекомендации по развитию

1. Держать все HTTP вызовы только в `lib/api.ts`.
2. Держать auth бизнес-логику только в `lib/authStore.ts`.
3. Для новых protected endpoint сразу подключать 401 handler.
4. Дублирующиеся interactive flow (page vs modal) поддерживать симметрично.
