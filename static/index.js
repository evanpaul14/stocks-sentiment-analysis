let chart = null;
let currentSymbol = null;
let historicalData = {};
let activeMovementInsightRequestId = 0;
const movementBox = document.getElementById('movementBox');
const movementSummaryEl = document.getElementById('movementSummary');
const movementMetaEl = document.getElementById('movementMeta');
const movementSourcesEl = document.getElementById('movementSources');
const trendingLists = {
    stocktwits: document.getElementById('stocktwitsList'),
    reddit: document.getElementById('redditList'),
    volume: document.getElementById('volumeList')
};
const TRENDING_SOURCES = ['stocktwits', 'reddit', 'volume'];
function normalizeTrendingSource(value) {
    const normalized = (value || '').toString().trim().toLowerCase();
    if (TRENDING_SOURCES.includes(normalized)) {
        return normalized;
    }
    return TRENDING_SOURCES[0];
}

function resolveTrendingSourceFromPath(pathname) {
    const match = pathname && pathname.match(/^\/trending-list(?:\/([a-z]+))?$/);
    if (match && match[1]) {
        return normalizeTrendingSource(match[1]);
    }
    return TRENDING_SOURCES[0];
}

function buildTrendingPath(source) {
    const normalized = normalizeTrendingSource(source);
    return `/trending-list/${normalized}`;
}

function queueIdleWork(callback, timeoutMs = 500) {
    if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
        window.requestIdleCallback(callback, { timeout: timeoutMs });
        return;
    }
    setTimeout(callback, 120);
}

const TRENDING_CACHE_TTL_MS = 5 * 60 * 1000;
const TRENDING_INCLUDE_PRICES = true;
const trendingCache = {};
const trendingFetchPromises = {};
const trendingPriceHydrations = new Map();
const loadedTrendingSources = new Set();
let trendingLazyObserver = null;
let trendingLazyInitialized = false;
const stocktwitsSummaryCache = new Map();
const stocktwitsSummaryPromises = new Map();
let stocktwitsSummaryPopover = null;
let stocktwitsPopoverHeading = null;
let stocktwitsPopoverBody = null;
let stocktwitsPopoverMeta = null;
const sentenceAbbreviations = new Set([
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'ltd', 'llc', 'plc',
    'sr', 'jr', 'dr', 'mr', 'mrs', 'ms', 'prof', 'dept', 'st'
]);
const WATCHLIST_STORAGE_KEY = 'ssa_watchlist_v1';
const RECENT_SEARCH_STORAGE_KEY = 'ssa_recent_searches_v1';
const MAX_RECENT_SEARCH_ITEMS = 8;
const companyInputEl = document.getElementById('companyInput');
const searchSuggestionsEl = document.getElementById('searchSuggestions');
const watchlistToggleBtn = document.getElementById('watchlistToggleBtn');
const watchlistListEl = document.getElementById('watchlistList');
const watchlistTableEl = document.getElementById('watchlistTable');
const watchlistEmptyStateEl = document.getElementById('watchlistEmptyState');
const watchlistSectionEl = document.getElementById('watchlistSection');
const marketSummarySectionEl = document.getElementById('marketSummarySection');
const marketSummaryLatestEl = document.getElementById('marketSummaryLatest');
const marketSummaryArchiveEl = document.getElementById('marketSummaryArchive');
const marketSummaryArchiveHeadingEl = document.getElementById('marketSummaryArchiveHeading');
const marketSummaryStatsRowsEl = document.getElementById('marketSummaryStatsRows');
const marketSummaryFormEl = document.getElementById('marketSummarySignupForm');
const marketSummaryEmailInputEl = document.getElementById('marketSummaryEmail');
const marketSummarySignupStatusEl = document.getElementById('marketSummarySignupStatus');
const marketSummarySignupButtonEl = document.getElementById('marketSummarySignupButton');
const marketSummarySlug = document.body.dataset.marketArticleSlug || '';
const marketSummaryBackBtnEl = document.getElementById('marketSummaryBackBtn');
const initialMarketSummaryArticleEl = document.getElementById('initialMarketSummaryArticle');
let initialMarketSummaryArticle = null;
if (initialMarketSummaryArticleEl) {
    try {
        initialMarketSummaryArticle = JSON.parse(initialMarketSummaryArticleEl.textContent);
    } catch (error) {
        console.error('Unable to parse initial market summary payload:', error);
    }
}
const pageVariant = document.body.dataset.page || 'home';
const initialSearchQuery = document.body.dataset.searchQuery || '';
const isHomePage = pageVariant === 'home';
const isWatchlistPage = pageVariant === 'watchlist';
const isTrendingPage = pageVariant === 'trending';
const isSearchPage = pageVariant === 'search';
const isMarketPage = pageVariant === 'market';
const homeTitleEl = document.getElementById('homeTitle');
const trendingBtnEl = document.getElementById('trendingBtn');
const watchlistNavBtnEl = document.getElementById('watchlistNavBtn');
const marketSummaryBtnEl = document.getElementById('marketSummaryBtn');
const unsplashReferralFallback = document.body.dataset.unsplashRef || 'https://unsplash.com/?utm_source=stocks-sentiment-analysis&utm_medium=referral';
let currentTrendingSource = normalizeTrendingSource(document.body.dataset.trendingSource || '');
let watchlistStorageAvailable = true;
let watchlistHasEntries = false;
let homePrimaryPanel = 'watchlist';
let homePanelInitialized = false;
let homeTrendingInitialized = false;
let watchlistPriceRefreshPromise = null;
let marketSummaryLatestArticleId = null;
let marketSummaryArchiveItems = [];
let activeSentimentRunId = 0;
let latestSentimentCounts = { positive: 0, negative: 0, neutral: 0 };
const sentimentStatusEl = document.getElementById('sentimentStatus');
const sentimentProgressEl = document.getElementById('sentimentProgress');
const sentimentProgressBarEl = sentimentProgressEl ? sentimentProgressEl.querySelector('.sentiment-progress-bar') : null;
let sentimentBatchSize = 0;
const SENTIMENT_REQUEST_TIMEOUT_MS = 25000;
const SENTIMENT_RETRY_TIMEOUT_MS = 45000;
let recentSearches = [];
let visibleSearchSuggestions = [];
let activeSearchSuggestionIndex = -1;

function initializeHomeScrollEffect() {
    if (!isHomePage || typeof window === 'undefined') return;
    const scrollBackground = document.querySelector('.home-scroll-bg');
    if (!scrollBackground) return;

    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return;
    }

    const docStyle = document.documentElement.style;
    let targetY = window.scrollY || window.pageYOffset || 0;
    let renderedY = targetY;
    let rafId = null;

    const renderFromY = (y) => {
        docStyle.setProperty('--home-shift-a', `${(y * 0.07).toFixed(2)}px`);
        docStyle.setProperty('--home-shift-b', `${(y * -0.05).toFixed(2)}px`);
        docStyle.setProperty('--home-shift-c', `${(y * 0.03).toFixed(2)}px`);
        docStyle.setProperty('--home-grid-shift', `${(y * -0.015).toFixed(2)}px`);
        docStyle.setProperty('--home-tilt', `${(y * 0.0015).toFixed(3)}deg`);
    };

    const animate = () => {
        const delta = targetY - renderedY;
        renderedY += delta * 0.12;
        if (Math.abs(delta) < 0.2) {
            renderedY = targetY;
        }
        renderFromY(renderedY);

        if (Math.abs(targetY - renderedY) >= 0.2) {
            rafId = window.requestAnimationFrame(animate);
        } else {
            rafId = null;
        }
    };

    const onScroll = () => {
        targetY = window.scrollY || window.pageYOffset || 0;
        if (rafId === null) {
            rafId = window.requestAnimationFrame(animate);
        }
    };

    renderFromY(targetY);
    window.addEventListener('scroll', onScroll, { passive: true });
}

function loadRecentSearches() {
    if (typeof localStorage === 'undefined') return [];
    try {
        const raw = localStorage.getItem(RECENT_SEARCH_STORAGE_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        return parsed
            .map(item => (item || '').toString().trim())
            .filter(Boolean)
            .slice(0, MAX_RECENT_SEARCH_ITEMS);
    } catch (error) {
        console.error('Unable to read recent searches:', error);
        return [];
    }
}

function saveRecentSearches() {
    if (typeof localStorage === 'undefined') return;
    try {
        localStorage.setItem(RECENT_SEARCH_STORAGE_KEY, JSON.stringify(recentSearches.slice(0, MAX_RECENT_SEARCH_ITEMS)));
    } catch (error) {
        console.error('Unable to save recent searches:', error);
    }
}

function addRecentSearch(query) {
    const normalized = (query || '').trim();
    if (!normalized) return;
    recentSearches = [
        normalized,
        ...recentSearches.filter(item => item.toLowerCase() !== normalized.toLowerCase())
    ].slice(0, MAX_RECENT_SEARCH_ITEMS);
    saveRecentSearches();
}

function getMatchingRecentSearches(query) {
    const normalized = (query || '').trim().toLowerCase();
    if (!normalized) {
        return recentSearches.slice(0, MAX_RECENT_SEARCH_ITEMS);
    }

    const startsWith = [];
    const includes = [];
    recentSearches.forEach((item) => {
        const lowered = item.toLowerCase();
        if (lowered.startsWith(normalized)) {
            startsWith.push(item);
        } else if (lowered.includes(normalized)) {
            includes.push(item);
        }
    });

    return [...startsWith, ...includes].slice(0, MAX_RECENT_SEARCH_ITEMS);
}

function setSearchSuggestionsExpandedState(expanded) {
    if (!companyInputEl) return;
    companyInputEl.setAttribute('aria-expanded', expanded ? 'true' : 'false');
}

function hideSearchSuggestions() {
    if (!searchSuggestionsEl) return;
    searchSuggestionsEl.classList.add('is-hidden');
    searchSuggestionsEl.innerHTML = '';
    visibleSearchSuggestions = [];
    activeSearchSuggestionIndex = -1;
    setSearchSuggestionsExpandedState(false);
}

function updateActiveSearchSuggestion() {
    if (!searchSuggestionsEl) return;
    const buttons = searchSuggestionsEl.querySelectorAll('.search-suggestion-item');
    buttons.forEach((button, idx) => {
        button.classList.toggle('is-active', idx === activeSearchSuggestionIndex);
    });
}

function handleSearchSuggestionSelect(value) {
    if (!companyInputEl) return;
    companyInputEl.value = value;
    hideSearchSuggestions();
    searchStock(value);
}

function renderSearchSuggestions(query = '') {
    if (!searchSuggestionsEl) return;

    const matches = getMatchingRecentSearches(query);
    if (!matches.length) {
        hideSearchSuggestions();
        return;
    }

    visibleSearchSuggestions = matches;
    activeSearchSuggestionIndex = -1;
    searchSuggestionsEl.innerHTML = `
        <div class="search-suggestions-header">Recent Searches</div>
        ${matches.map(item => `
            <button type="button" class="search-suggestion-item" role="option" data-value="${escapeAttribute(item)}">
                <span class="search-suggestion-value">${escapeHtml(item)}</span>
                <span class="search-suggestion-meta">Use Search</span>
            </button>
        `).join('')}
    `;

    searchSuggestionsEl.classList.remove('is-hidden');
    setSearchSuggestionsExpandedState(true);

    const buttons = searchSuggestionsEl.querySelectorAll('.search-suggestion-item');
    buttons.forEach((button) => {
        button.addEventListener('mousedown', (event) => {
            event.preventDefault();
        });
        button.addEventListener('click', () => {
            handleSearchSuggestionSelect(button.dataset.value || '');
        });
    });
}

function initializeSearchSuggestions() {
    if (!companyInputEl || !searchSuggestionsEl) return;
    recentSearches = loadRecentSearches();

    companyInputEl.addEventListener('focus', () => {
        renderSearchSuggestions(companyInputEl.value);
    });

    companyInputEl.addEventListener('input', () => {
        renderSearchSuggestions(companyInputEl.value);
    });

    companyInputEl.addEventListener('keydown', (event) => {
        if (!visibleSearchSuggestions.length) return;

        if (event.key === 'ArrowDown') {
            event.preventDefault();
            activeSearchSuggestionIndex = (activeSearchSuggestionIndex + 1) % visibleSearchSuggestions.length;
            updateActiveSearchSuggestion();
            return;
        }

        if (event.key === 'ArrowUp') {
            event.preventDefault();
            activeSearchSuggestionIndex = (activeSearchSuggestionIndex - 1 + visibleSearchSuggestions.length) % visibleSearchSuggestions.length;
            updateActiveSearchSuggestion();
            return;
        }

        if (event.key === 'Enter' && activeSearchSuggestionIndex >= 0) {
            event.preventDefault();
            handleSearchSuggestionSelect(visibleSearchSuggestions[activeSearchSuggestionIndex]);
            return;
        }

        if (event.key === 'Escape') {
            hideSearchSuggestions();
        }
    });

    document.addEventListener('click', (event) => {
        if (!event.target.closest || !event.target.closest('.search-wrap')) {
            hideSearchSuggestions();
        }
    });
}

    function formatNumber(num) {
        if (num >= 1e12) return (num / 1e12).toFixed(2) + 'T';
        if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
        if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
        if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
        return num.toFixed(2);
    }

    function escapeAttribute(value) {
        if (value === undefined || value === null) return '';
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;');
    }

    function escapeHtml(value) {
        if (value === undefined || value === null) return '';
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function formatCurrency(num) {
        if (!Number.isFinite(num)) return '$0.00';
        const sign = num < 0 ? '-' : '';
        return sign + '$' + Math.abs(num).toFixed(2);
    }

    function formatPercentage(num) {
        return num.toFixed(2) + '%';
    }

    function formatSignedPercentage(num) {
        if (!Number.isFinite(num)) return '0.00%';
        const sign = num >= 0 ? '+' : '';
        return `${sign}${num.toFixed(2)}%`;
    }

    function formatOptionalCurrency(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return null;
        return '$' + num.toFixed(2);
    }

    function formatTrendPercent(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return null;
        const sign = num > 0 ? '+' : '';
        return `${sign}${num.toFixed(1)}%`;
    }

    function buildTrendChangeBadge(value, extraClass = '') {
        const num = Number(value);
        if (!Number.isFinite(num)) return '';
        const classes = ['ticker-change'];
        if (Object.is(num, 0)) {
            classes.push('neutral');
        } else if (num > 0) {
            classes.push('up');
        } else {
            classes.push('down');
        }
        if (extraClass) classes.push(extraClass);
        return `<span class="${classes.join(' ')}">${formatSignedPercentage(num)}</span>`;
    }

    function updateTrendingChangeBadges(itemEl, changeValue) {
        if (!itemEl) return;
        const inlineTarget = itemEl.querySelector('[data-change-inline]');
        const rightTarget = itemEl.querySelector('[data-change-right]');
        const inlineMarkup = buildTrendChangeBadge(changeValue, 'trending-change-inline');
        if (inlineTarget) {
            inlineTarget.innerHTML = inlineMarkup || '';
        }
        const rightMarkup = buildTrendChangeBadge(changeValue, 'trending-change-right');
        if (rightTarget) {
            rightTarget.innerHTML = rightMarkup || '';
        }
    }

    function findTrendingItemElement(source, ticker) {
        const list = trendingLists[source];
        if (!list || !ticker) return null;
        const normalized = ticker.toUpperCase();
        return Array.from(list.querySelectorAll('.ticker-row, .trending-item')).find(item => {
            return (item.dataset.ticker || '').toUpperCase() === normalized;
        }) || null;
    }

    function applyTrendingPriceUpdate(source, ticker, changePercent) {
        if (!ticker) return;
        const targetItem = findTrendingItemElement(source, ticker);
        if (!targetItem) return;
        updateTrendingChangeBadges(targetItem, changePercent);
    }

    function hydrateTrendingPrices(source, items) {
        if (!Array.isArray(items) || !items.length) return null;
        if (!trendingLists[source]) return null;

        const tickers = Array.from(new Set(
            items.slice(0, 12)
                .map(item => (item.ticker || '').toUpperCase())
                .filter(Boolean)
        ));
        if (!tickers.length) return null;

        const hydrationKey = `${source}:${tickers.join(',')}`;
        if (trendingPriceHydrations.has(hydrationKey)) {
            return trendingPriceHydrations.get(hydrationKey);
        }

        const hydrationPromise = (async () => {
            const quotePromises = tickers.map(async ticker => {
                try {
                    const quote = await fetchQuote(ticker);
                    if (quote && typeof quote.change_percent === 'number') {
                        applyTrendingPriceUpdate(source, ticker, Number(quote.change_percent));
                    }
                } catch (error) {
                    console.error('Trending price hydrate error:', ticker, error);
                }
            });
            await Promise.allSettled(quotePromises);
        })().finally(() => {
            trendingPriceHydrations.delete(hydrationKey);
        });

        trendingPriceHydrations.set(hydrationKey, hydrationPromise);
        return hydrationPromise;
    }

    function formatRelativeTimeFromISOString(isoString) {
        if (!isoString) return '';
        const date = new Date(isoString);
        if (Number.isNaN(date.getTime())) return '';
        const diffMinutes = Math.max(0, Math.floor((Date.now() - date.getTime()) / 60000));
        if (diffMinutes < 1) return 'just now';
        if (diffMinutes < 60) return `${diffMinutes}m ago`;
        const diffHours = Math.floor(diffMinutes / 60);
        if (diffHours < 24) return `${diffHours}h ago`;
        const diffDays = Math.floor(diffHours / 24);
        return `${diffDays}d ago`;
    }

    function formatMarketSummaryTimestamp(isoString) {
        if (!isoString) return 'Pending release';
        const date = new Date(isoString);
        if (Number.isNaN(date.getTime())) return 'Pending release';
        // Detect US locale, use MM/DD/YYYY; else DD/MM/YYYY
        const locale = navigator.language || 'en';
        const isUS = locale.startsWith('en-US');
        const y = date.getFullYear();
        const m = String(date.getMonth() + 1).padStart(2, '0');
        const d = String(date.getDate()).padStart(2, '0');
        let hour = date.getHours();
        const min = String(date.getMinutes()).padStart(2, '0');
        const ampm = hour >= 12 ? 'PM' : 'AM';
        hour = hour % 12;
        if (hour === 0) hour = 12;
        const hourStr = String(hour).padStart(2, '0');
        const dateStr = isUS ? `${m}/${d}/${y}` : `${d}/${m}/${y}`;
        return `${dateStr}, ${hourStr}:${min} ${ampm}`;
    }

    function parseMarketSummaryDate(rawValue) {
        if (!rawValue) return null;
        if (rawValue instanceof Date) {
            return Number.isNaN(rawValue.getTime()) ? null : rawValue;
        }
        const text = String(rawValue).trim();
        if (!text) return null;

        // Parse date-only fields (YYYY-MM-DD) as local calendar dates to avoid timezone shifts.
        const dateOnlyMatch = text.match(/^(\d{4})-(\d{2})-(\d{2})$/);
        if (dateOnlyMatch) {
            const year = Number(dateOnlyMatch[1]);
            const month = Number(dateOnlyMatch[2]);
            const day = Number(dateOnlyMatch[3]);
            const localDate = new Date(year, month - 1, day);
            if (!Number.isNaN(localDate.getTime())) {
                return localDate;
            }
        }

        const parsed = new Date(text);
        return Number.isNaN(parsed.getTime()) ? null : parsed;
    }

    function resolveMarketSummaryDate(record) {
        if (!record) return null;
        return (
            parseMarketSummaryDate(record.summary_date)
            || parseMarketSummaryDate(record.published_at)
            || parseMarketSummaryDate(record.created_at)
        );
    }

    function toLocalMidnight(date) {
        if (!(date instanceof Date) || Number.isNaN(date.getTime())) return null;
        return new Date(date.getFullYear(), date.getMonth(), date.getDate());
    }

    function buildMarketSummaryDateMeta(record, { isLatest = false, referenceDate = null } = {}) {
        const parsed = resolveMarketSummaryDate(record);
        if (!parsed) {
            return {
                context: isLatest ? 'Latest edition' : 'Recent edition',
                dayName: 'Market Summary',
                dayDate: 'Date unavailable'
            };
        }

        const contextBase = isLatest ? 'Latest edition' : 'Recent edition';
        let context = contextBase;
        const localParsed = toLocalMidnight(parsed);
        const localReference = toLocalMidnight(referenceDate);

        if (localParsed && localReference) {
            const diffMs = localReference.getTime() - localParsed.getTime();
            const diffDays = Math.round(diffMs / (1000 * 60 * 60 * 24));
            if (diffDays === 0) {
                context = 'Today';
            } else if (diffDays === 1) {
                context = 'Yesterday';
            } else if (diffDays > 1) {
                context = `${diffDays} days ago`;
            }
        }

        return {
            context,
            dayName: parsed.toLocaleDateString(undefined, { weekday: 'long' }),
            dayDate: parsed.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
        };
    }

    function renderMarketSummaryStats(items) {
        if (!marketSummaryStatsRowsEl) return;
        const source = Array.isArray(items) ? items : [];

        const summaryTargets = [
            { label: 'S&P 500', aliases: ['s&p 500', 'sp500', 'spx', '^gspc', 'gspc'], symbols: ['^gspc', 'gspc', 'spx'] },
            { label: 'Nasdaq', aliases: ['nasdaq', 'nasdaq composite', 'ixic', '^ixic', 'nasdaq 100', 'ndx'], symbols: ['^ixic', 'ixic', 'ndx'] },
            { label: 'Dow Jones', aliases: ['dow jones', 'dow', 'djia', '^dji', 'dji'], symbols: ['^dji', 'dji', 'djia'] }
        ];

        const normalize = (value) => String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
        const normalizeCompact = (value) => normalize(value).replace(/\s+/g, '');
        const parseMetricNumber = (value) => {
            if (typeof value === 'number') return Number.isFinite(value) ? value : NaN;
            if (typeof value === 'string') {
                const cleaned = value.replace(/[%,$\s]/g, '');
                const parsed = Number(cleaned);
                return Number.isFinite(parsed) ? parsed : NaN;
            }
            const parsed = Number(value);
            return Number.isFinite(parsed) ? parsed : NaN;
        };

        const findMatch = (aliases, symbols) => {
            const normalizedAliases = aliases.map(alias => normalizeCompact(alias));
            const normalizedSymbols = symbols.map(symbol => normalizeCompact(symbol));

            // Prefer exact symbol match when week_glance payload includes canonical index symbols.
            const symbolHit = source.find(item => {
                const symbol = normalizeCompact(item && item.symbol);
                return symbol && normalizedSymbols.includes(symbol);
            });
            if (symbolHit) return symbolHit;

            return source.find(item => {
                const haystack = normalizeCompact(`${item && item.label} ${item && item.symbol}`);
                return normalizedAliases.some(alias => alias && haystack.includes(alias));
            }) || null;
        };

        marketSummaryStatsRowsEl.innerHTML = '';
        summaryTargets.forEach(target => {
            const match = findMatch(target.aliases, target.symbols);
            const rawChange = parseMetricNumber(match && (match.week_change_percent ?? match.change_percent));
            const rawClose = parseMetricNumber(match && match.close);
            let valueText = '--';
            let valueClass = 'stat-value';

            if (Number.isFinite(rawChange)) {
                valueText = formatSignedPercentage(rawChange);
                valueClass += rawChange >= 0 ? ' stat-value-positive' : ' stat-value-negative';
            } else if (Number.isFinite(rawClose)) {
                valueText = formatCurrency(rawClose);
            }

            const row = document.createElement('div');
            row.className = 'stat-row';
            row.innerHTML = `
                <span class="stat-label">${escapeHtml(target.label)}</span>
                <span class="${valueClass}">${escapeHtml(valueText)}</span>
            `;
            marketSummaryStatsRowsEl.appendChild(row);
        });
    }

    function renderSentimentSummaryUI(counts, completedTotal = 0, loading = false) {
        updateSentimentProgress(completedTotal, Boolean(loading));
        const positive = counts.positive || 0;
        const negative = counts.negative || 0;
        const neutral = counts.neutral || 0;
        const total = completedTotal || (positive + negative + neutral);

        const summaryEl = document.querySelector('.sentiment-summary');
        if (summaryEl) {
            summaryEl.classList.toggle('is-loading', Boolean(loading));
        }

        function percentValue(val) {
            if (!total) return 0;
            return Math.round((val / total) * 100);
        }

        function percentText(val) {
            if (!total) return loading ? '...' : '0%';
            return `${percentValue(val)}%`;
        }

        const positivePct = percentValue(positive);
        const negativePct = percentValue(negative);
        const neutralPct = Math.max(0, 100 - positivePct - negativePct);

        document.getElementById('positiveCount').textContent = percentText(positive);
        document.getElementById('negativeCount').textContent = percentText(negative);
        document.getElementById('neutralCount').textContent = percentText(neutral);

        const positiveBar = document.getElementById('positiveBar');
        const neutralBar = document.getElementById('neutralBar');
        const negativeBar = document.getElementById('negativeBar');
        if (positiveBar) positiveBar.style.width = `${positivePct}%`;
        if (neutralBar) neutralBar.style.width = `${neutralPct}%`;
        if (negativeBar) negativeBar.style.width = `${negativePct}%`;

        const donutPrimary = document.getElementById('sentimentDonutPrimary');
        const donutSecondary = document.getElementById('sentimentDonutSecondary');
        const positiveArc = document.getElementById('positiveArc');
        const neutralArc = document.getElementById('neutralArc');
        const negativeArc = document.getElementById('negativeArc');
        const circumference = 2 * Math.PI * 30;
        const positiveArcLen = (positivePct / 100) * circumference;
        const neutralArcLen = (neutralPct / 100) * circumference;
        const negativeArcLen = (negativePct / 100) * circumference;
        if (positiveArc) {
            positiveArc.setAttribute('stroke-dasharray', `${positiveArcLen} ${circumference}`);
            positiveArc.setAttribute('stroke-dashoffset', '0');
        }
        if (neutralArc) {
            neutralArc.setAttribute('stroke-dasharray', `${neutralArcLen} ${circumference}`);
            neutralArc.setAttribute('stroke-dashoffset', `${-positiveArcLen}`);
        }
        if (negativeArc) {
            negativeArc.setAttribute('stroke-dasharray', `${negativeArcLen} ${circumference}`);
            negativeArc.setAttribute('stroke-dashoffset', `${-(positiveArcLen + neutralArcLen)}`);
        }

        const overallElement = document.getElementById('overallSentiment');
        let overall = '—';
        let overallClass = 'neutral';
        if (total > 0) {
            const entries = [
                ['positive', positive],
                ['negative', negative],
                ['neutral', neutral]
            ];
            entries.sort((a, b) => b[1] - a[1]);
            overall = entries[0][0];
            overallClass = overall;
        } else if (loading) {
            overall = '…';
        }
        const overallLabelMap = {
            positive: '↑ Overall Bullish',
            negative: '↓ Overall Bearish',
            neutral: '→ Overall Neutral'
        };
        const overallLabel = total > 0 ? (overallLabelMap[overall] || '→ Overall Neutral') : (loading ? 'Analyzing…' : 'No sentiment');
        overallElement.textContent = overallLabel;
        overallElement.className = `overall-sent sentiment-value ${overallClass}`;

        if (donutPrimary) {
            donutPrimary.textContent = total > 0 ? `${Math.max(positivePct, neutralPct, negativePct)}%` : '0%';
        }
        if (donutSecondary) {
            donutSecondary.textContent = total > 0 ? (overall || 'neutral') : 'neutral';
        }
    }

    function resetSentimentSummaryUI() {
        latestSentimentCounts = { positive: 0, negative: 0, neutral: 0 };
        sentimentBatchSize = 0;
        updateSentimentProgress(0, false);
        renderSentimentSummaryUI(latestSentimentCounts, 0, true);
    }

    function setSentimentStatus(message, tone = 'default') {
        if (!sentimentStatusEl) return;
        const shouldHide = !message;
        sentimentStatusEl.style.display = shouldHide ? 'none' : 'block';
        if (shouldHide) {
            sentimentStatusEl.textContent = '';
            return;
        }
        sentimentStatusEl.textContent = message;
        sentimentStatusEl.style.color = tone === 'error' ? '#ef4444' : '#a1a1aa';
    }

    function setMarketSummarySignupStatus(message, tone = 'default') {
        if (!marketSummarySignupStatusEl) return;
        marketSummarySignupStatusEl.textContent = message || '';
        marketSummarySignupStatusEl.classList.remove('success', 'error');
        if (!message) {
            return;
        }
        if (tone === 'success') {
            marketSummarySignupStatusEl.classList.add('success');
        } else if (tone === 'error') {
            marketSummarySignupStatusEl.classList.add('error');
        }
    }

    async function handleMarketSummarySignup(event) {
        event.preventDefault();
        if (!marketSummaryEmailInputEl || !marketSummarySignupButtonEl) return;
        const email = marketSummaryEmailInputEl.value.trim();
        if (!email) {
            setMarketSummarySignupStatus('Enter your email to subscribe.', 'error');
            marketSummaryEmailInputEl.focus();
            return;
        }
        setMarketSummarySignupStatus('Signing you up...');
        marketSummarySignupButtonEl.disabled = true;
        try {
            const res = await fetch('/api/market-summary/subscribe', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email })
            });
            const payload = await res.json();
            if (!res.ok) {
                throw new Error(payload && payload.error ? payload.error : 'Unable to subscribe.');
            }
            setMarketSummarySignupStatus(payload.message || 'Check your inbox to confirm.', 'success');
            marketSummaryEmailInputEl.value = '';
        } catch (error) {
            const fallbackMessage = (error && error.message) ? error.message : 'Unable to subscribe right now.';
            setMarketSummarySignupStatus(fallbackMessage, 'error');
        } finally {
            marketSummarySignupButtonEl.disabled = false;
        }
    }

    function updateSentimentProgress(completed = 0, loading = false) {
        if (!sentimentProgressEl || !sentimentProgressBarEl) return;
        if (!loading || sentimentBatchSize <= 0) {
            sentimentProgressEl.classList.remove('is-active');
            sentimentProgressEl.setAttribute('aria-hidden', 'true');
            sentimentProgressBarEl.style.width = '0%';
            return;
        }
        const ratio = sentimentBatchSize ? Math.min(1, completed / sentimentBatchSize) : 0;
        sentimentProgressEl.classList.add('is-active');
        sentimentProgressEl.setAttribute('aria-hidden', 'false');
        sentimentProgressBarEl.style.width = `${(ratio * 100).toFixed(2)}%`;
    }

    function createParagraphNodes(container, text) {
        const bodyText = (text || '').split(/\n+/).map(p => p.trim()).filter(Boolean);
        if (!bodyText.length) {
            const fallback = document.createElement('p');
            fallback.textContent = 'Summary coming soon.';
            container.appendChild(fallback);
            return;
        }
        bodyText.forEach(chunk => {
            const p = document.createElement('p');
            p.textContent = chunk;
            container.appendChild(p);
        });
    }

    function readWatchlist() {
        if (!watchlistStorageAvailable || typeof localStorage === 'undefined') {
            return [];
        }
        try {
            const raw = localStorage.getItem(WATCHLIST_STORAGE_KEY);
            if (!raw) return [];
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
                return parsed.filter(item => item && item.symbol);
            }
        } catch (error) {
            console.error('Watchlist storage error:', error);
            watchlistStorageAvailable = false;
        }
        return [];
    }

    function writeWatchlist(items) {
        if (!watchlistStorageAvailable || typeof localStorage === 'undefined') return;
        try {
            localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(items));
        } catch (error) {
            console.error('Unable to persist watchlist:', error);
            watchlistStorageAvailable = false;
        }
    }

    function isSymbolInWatchlist(symbol) {
        if (!symbol) return false;
        const normalized = symbol.toUpperCase();
        return readWatchlist().some(item => item.symbol === normalized);
    }

    function addSymbolToWatchlist(entry) {
        if (!entry || !entry.symbol || !watchlistStorageAvailable) return;
        const normalized = entry.symbol.toUpperCase();
        const items = readWatchlist();
        if (items.some(item => item.symbol === normalized)) return;
        const nowIso = new Date().toISOString();
        const newEntry = {
            symbol: normalized,
            companyName: entry.companyName || normalized,
            lastPrice: Number.isFinite(entry.lastPrice) ? Number(entry.lastPrice) : null,
            lastUpdated: Number.isFinite(entry.lastPrice) ? nowIso : null,
            addedAt: nowIso
        };
        const nextItems = [newEntry, ...items].slice(0, 50);
        writeWatchlist(nextItems);
        renderWatchlist();
    }

    function removeSymbolFromWatchlist(symbol) {
        if (!symbol || !watchlistStorageAvailable) return;
        const normalized = symbol.toUpperCase();
        const filtered = readWatchlist().filter(item => item.symbol !== normalized);
        writeWatchlist(filtered);
        renderWatchlist();
    }

    function updateWatchlistSnapshot(symbol, price, companyName) {
        if (!symbol || !watchlistStorageAvailable || !isSymbolInWatchlist(symbol)) return;
        const normalized = symbol.toUpperCase();
        const items = readWatchlist();
        let changed = false;
        const updatedItems = items.map(item => {
            if (item.symbol !== normalized) return item;
            const updated = { ...item };
            if (companyName) {
                updated.companyName = companyName;
            }
            if (Number.isFinite(price)) {
                updated.lastPrice = Number(price);
                updated.lastUpdated = new Date().toISOString();
            }
            changed = true;
            return updated;
        });
        if (changed) {
            writeWatchlist(updatedItems);
            renderWatchlist();
        }
    }

    function updateWatchlistToggle(symbol, companyName, price) {
        if (!watchlistToggleBtn) return;
        if (!symbol) {
            watchlistToggleBtn.style.display = 'none';
            watchlistToggleBtn.classList.add('is-hidden');
            watchlistToggleBtn.removeAttribute('data-symbol');
            return;
        }
        const normalized = symbol.toUpperCase();
        watchlistToggleBtn.style.display = 'inline-flex';
        watchlistToggleBtn.classList.remove('is-hidden');
        watchlistToggleBtn.dataset.symbol = normalized;
        watchlistToggleBtn.dataset.company = companyName || normalized;
        if (Number.isFinite(price)) {
            watchlistToggleBtn.dataset.price = Number(price);
        } else {
            delete watchlistToggleBtn.dataset.price;
        }
        const saved = isSymbolInWatchlist(normalized);
        watchlistToggleBtn.textContent = saved ? 'Remove from Watchlist' : 'Add to Watchlist';
        watchlistToggleBtn.classList.toggle('saved', saved);
        watchlistToggleBtn.setAttribute('aria-pressed', saved ? 'true' : 'false');
    }

    function renderWatchlist() {
        if (!watchlistListEl) return 0;
        const items = readWatchlist();
        watchlistHasEntries = items.length > 0;
        if (!items.length) {
            watchlistListEl.innerHTML = '';
            if (watchlistTableEl) {
                watchlistTableEl.classList.add('is-hidden');
            }
            if (watchlistEmptyStateEl) {
                watchlistEmptyStateEl.style.display = 'block';
            }
            if (isWatchlistPage) {
                watchlistSectionEl.style.display = '';
            } else if (!isHomePage) {
                showWatchlistSection(false);
            }
            if (isHomePage) {
                setHomePrimaryPanel('watchlist');
            }
            return 0;
        }
        if (watchlistEmptyStateEl) {
            watchlistEmptyStateEl.style.display = 'none';
        }
        if (watchlistTableEl) {
            watchlistTableEl.classList.remove('is-hidden');
        }
        watchlistListEl.innerHTML = '';
        items.forEach(item => {
            const row = document.createElement('tr');
            row.className = 'watchlist-row';
            row.title = `${item.symbol} · Click to load`;
            row.addEventListener('click', () => {
                document.getElementById('companyInput').value = item.symbol;
                searchStock();
            });

            const symbolCell = document.createElement('td');
            symbolCell.dataset.label = 'Symbol';
            const left = document.createElement('div');
            left.className = 'ticker-cell';
            const logo = document.createElement('div');
            logo.className = 'ticker-logo';
            logo.textContent = item.symbol.charAt(0);
            const symbolEl = document.createElement('div');
            symbolEl.className = 'ticker-text-sym';
            symbolEl.textContent = item.symbol;
            const nameEl = document.createElement('div');
            nameEl.className = 'ticker-text-name';
            nameEl.textContent = item.companyName || '';
            const labelWrap = document.createElement('div');
            labelWrap.appendChild(symbolEl);
            labelWrap.appendChild(nameEl);
            left.appendChild(logo);
            left.appendChild(labelWrap);
            symbolCell.appendChild(left);

            const priceCell = document.createElement('td');
            priceCell.dataset.label = 'Price';
            if (Number.isFinite(item.lastPrice)) {
                const priceEl = document.createElement('div');
                priceEl.className = 'price-val';
                priceEl.textContent = formatCurrency(item.lastPrice);
                priceCell.appendChild(priceEl);
            }

            const updatedCell = document.createElement('td');
            updatedCell.dataset.label = 'Updated';
            const updatedEl = document.createElement('div');
            updatedEl.className = 'watchlist-updated';
            updatedEl.textContent = item.lastUpdated
                ? `Updated ${formatRelativeTimeFromISOString(item.lastUpdated)}`
                : 'Tap to refresh';
            updatedCell.appendChild(updatedEl);

            const removeCell = document.createElement('td');
            removeCell.dataset.label = 'Actions';
            const removeBtn = document.createElement('button');
            removeBtn.type = 'button';
            removeBtn.className = 'watchlist-remove-btn';
            removeBtn.setAttribute('aria-label', `Remove ${item.symbol} from watchlist`);
            removeBtn.innerHTML = '&times;';
            removeBtn.addEventListener('click', event => {
                event.stopPropagation();
                removeSymbolFromWatchlist(item.symbol);
                if (watchlistToggleBtn && watchlistToggleBtn.dataset.symbol === item.symbol) {
                    updateWatchlistToggle(item.symbol, item.companyName, Number(watchlistToggleBtn.dataset.price));
                }
            });
            removeCell.appendChild(removeBtn);

            row.appendChild(symbolCell);
            row.appendChild(priceCell);
            row.appendChild(updatedCell);
            row.appendChild(removeCell);
            watchlistListEl.appendChild(row);
        });
        if (isWatchlistPage) {
            watchlistSectionEl.style.display = '';
        } else if (!isHomePage) {
            showWatchlistSection(homePrimaryPanel === 'watchlist');
        }
        if (isHomePage) {
            setHomePrimaryPanel('watchlist');
        }
        return items.length;
    }

    function handleWatchlistToggleClick() {
        if (!watchlistToggleBtn || !watchlistStorageAvailable) return;
        const symbol = watchlistToggleBtn.dataset.symbol;
        if (!symbol) return;
        const companyName = watchlistToggleBtn.dataset.company || symbol;
        const price = Number(watchlistToggleBtn.dataset.price);
        if (isSymbolInWatchlist(symbol)) {
            removeSymbolFromWatchlist(symbol);
        } else {
            addSymbolToWatchlist({
                symbol,
                companyName,
                lastPrice: Number.isFinite(price) ? price : null
            });
        }
        updateWatchlistToggle(symbol, companyName, price);
    }

    function getFirstSentence(text) {
        if (!text) return '';
        const normalized = text.replace(/\s+/g, ' ').trim();
        if (!normalized) return '';

        for (let i = 0; i < normalized.length; i++) {
            const char = normalized[i];
            if (!'.!?'.includes(char)) continue;

            const before = normalized.slice(0, i).trimEnd();
            const wordMatch = before.match(/([A-Za-z]+)$/);
            const lastWord = wordMatch ? wordMatch[1].toLowerCase() : '';
            const remainder = normalized.slice(i + 1);
            const hasMoreContent = /\S/.test(remainder);

            if (char === '.' && lastWord && sentenceAbbreviations.has(lastWord) && hasMoreContent) {
                continue;
            }

            const sentence = normalized.slice(0, i + 1).trim();
            if (sentence.length) {
                return sentence;
            }
        }

        const newlineSplit = text.split(/\n+/).map(part => part.replace(/\s+/g, ' ').trim()).filter(Boolean);
        if (newlineSplit.length > 0) {
            return newlineSplit[0];
        }

        return normalized;
    }

    function setTrendingListLoading(source) {
        const list = trendingLists[source];
        if (list) {
            list.innerHTML = '<div class="trending-message loading-state"><span class="loading-dot"></span><span>Gathering trending data...</span></div>';
        }
    }

    function showTrendingMessage(source, message, isError = false) {
        const list = trendingLists[source];
        if (!list) return;
        list.innerHTML = '';
        const msgEl = document.createElement('div');
        msgEl.className = `trending-message ${isError ? 'error' : ''}`;
        msgEl.textContent = message || 'No data available.';
        list.appendChild(msgEl);
    }

    function setMarketSummaryLoadingState(active, message = 'Loading market summary...') {
        if (!isMarketPage) return;

        if (marketSummaryBackBtnEl) {
            if (marketSummarySlug) {
                marketSummaryBackBtnEl.classList.remove('is-hidden');
            } else {
                marketSummaryBackBtnEl.classList.add('is-hidden');
            }
        }

        if (!active) {
            return;
        }

        const loadingMarkup = `<div class="market-summary-empty loading-state"><span class="loading-dot"></span><span>${escapeHtml(message)}</span></div>`;

        if (marketSummaryLatestEl) {
            marketSummaryLatestEl.innerHTML = loadingMarkup;
        }

        if (marketSummaryArchiveEl) {
            if (marketSummarySlug) {
                marketSummaryArchiveEl.innerHTML = '';
                marketSummaryArchiveEl.style.display = 'none';
            } else {
                marketSummaryArchiveEl.style.display = '';
                marketSummaryArchiveEl.innerHTML = loadingMarkup;
            }
        }

        if (marketSummaryStatsRowsEl) {
            marketSummaryStatsRowsEl.innerHTML = '<div class="market-summary-empty loading-state"><span class="loading-dot"></span><span>Loading week at a glance...</span></div>';
        }
    }

    function setMarketSummaryLatestLoading(message = 'Loading latest market summary...') {
        if (!marketSummaryLatestEl) return;
        marketSummaryLatestEl.innerHTML = `<div class="market-summary-empty loading-state"><span class="loading-dot"></span><span>${escapeHtml(message)}</span></div>`;
    }

    function setMarketSummaryArchiveLoading(message = 'Loading archive...') {
        if (!marketSummaryArchiveEl || marketSummarySlug) return;
        marketSummaryArchiveEl.style.display = '';
        marketSummaryArchiveEl.innerHTML = `<div class="market-summary-empty loading-state"><span class="loading-dot"></span><span>${escapeHtml(message)}</span></div>`;
    }

    function setMarketSummaryStatsLoading(message = 'Loading week at a glance...') {
        if (!marketSummaryStatsRowsEl) return;
        marketSummaryStatsRowsEl.innerHTML = `<div class="market-summary-empty loading-state"><span class="loading-dot"></span><span>${escapeHtml(message)}</span></div>`;
    }

    function renderMarketSummaryLatestError(message) {
        if (!marketSummaryLatestEl) return;
        marketSummaryLatestEl.innerHTML = '';
        const msgEl = document.createElement('div');
        msgEl.className = 'market-summary-empty error';
        msgEl.textContent = message || 'Unable to load market summary.';
        marketSummaryLatestEl.appendChild(msgEl);
    }

    function renderMarketSummaryArchiveError(message) {
        if (!marketSummaryArchiveEl || marketSummarySlug) return;
        marketSummaryArchiveEl.innerHTML = '';
        const msgEl = document.createElement('div');
        msgEl.className = 'market-summary-empty error';
        msgEl.textContent = message || 'Unable to load past editions.';
        marketSummaryArchiveEl.appendChild(msgEl);
    }

    function syncMarketSummaryArchive() {
        if (marketSummarySlug || !marketSummaryArchiveEl) {
            return;
        }
        renderMarketSummaryArchive(marketSummaryArchiveItems, marketSummaryLatestArticleId);
    }

    function isTrendingCacheFresh(entry, includePrices = true) {
        if (!entry) return false;
        if ((Date.now() - entry.timestamp) >= TRENDING_CACHE_TTL_MS) {
            return false;
        }
        if (includePrices && !entry.includePrices) {
            return false;
        }
        return true;
    }

    function getFreshTrendingCache(source, includePrices = true) {
        const entry = trendingCache[source];
        return isTrendingCacheFresh(entry, includePrices) ? entry : null;
    }

    function setTrendingCache(source, items, includePrices = true) {
        trendingCache[source] = {
            data: Array.isArray(items) ? items : [],
            timestamp: Date.now(),
            includePrices: Boolean(includePrices)
        };
    }

    function ensureStocktwitsSummaryPopover() {
        if (stocktwitsSummaryPopover) {
            return stocktwitsSummaryPopover;
        }
        const popover = document.createElement('div');
        popover.className = 'trending-summary-popover';
        popover.setAttribute('role', 'status');
        popover.innerHTML = `
            <div class="trending-summary-popover-heading" data-role="heading"></div>
            <div class="trending-summary-popover-body" data-role="body"></div>
            <div class="trending-summary-popover-meta" data-role="meta"></div>
        `;
        document.body.appendChild(popover);
        stocktwitsSummaryPopover = popover;
        stocktwitsPopoverHeading = popover.querySelector('[data-role="heading"]');
        stocktwitsPopoverBody = popover.querySelector('[data-role="body"]');
        stocktwitsPopoverMeta = popover.querySelector('[data-role="meta"]');
        return popover;
    }

    function hideStocktwitsPopover() {
        if (!stocktwitsSummaryPopover) return;
        stocktwitsSummaryPopover.classList.remove('visible', 'is-loading', 'is-error');
        stocktwitsSummaryPopover.style.display = 'none';
        stocktwitsSummaryPopover.removeAttribute('data-ticker');
    }

    function positionStocktwitsPopover(anchorEl) {
        const popover = ensureStocktwitsSummaryPopover();
        if (!anchorEl) return;
        popover.style.visibility = 'hidden';
        popover.style.display = 'block';
        const anchorRect = anchorEl.getBoundingClientRect();
        const popRect = popover.getBoundingClientRect();
        const top = window.scrollY + anchorRect.bottom + 10;
        const minLeft = window.scrollX + 12;
        const maxLeft = window.scrollX + window.innerWidth - popRect.width - 12;
        let left = window.scrollX + anchorRect.left - (popRect.width / 2) + (anchorRect.width / 2);
        left = Math.min(Math.max(left, minLeft), Math.max(minLeft, maxLeft));
        popover.style.top = `${top}px`;
        popover.style.left = `${left}px`;
        const anchorCenter = window.scrollX + anchorRect.left + (anchorRect.width / 2);
        const offsetWithinPopover = anchorCenter - left;
        const arrowOffset = Math.min(Math.max(offsetWithinPopover, 16), popRect.width - 16);
        popover.style.setProperty('--popover-arrow-offset', `${arrowOffset}px`);
        popover.style.visibility = 'visible';
    }

    function setStocktwitsPopoverState({ heading, body, meta, modifier }) {
        const popover = ensureStocktwitsSummaryPopover();
        popover.classList.remove('is-loading', 'is-error');
        if (modifier === 'loading') {
            popover.classList.add('is-loading');
        } else if (modifier === 'error') {
            popover.classList.add('is-error');
        }
        if (stocktwitsPopoverHeading) {
            stocktwitsPopoverHeading.textContent = heading || '';
        }
        if (stocktwitsPopoverBody) {
            stocktwitsPopoverBody.textContent = body || '';
        }
        if (stocktwitsPopoverMeta) {
            stocktwitsPopoverMeta.textContent = meta || '';
        }
    }

    function showStocktwitsPopover(anchorEl, ticker, state) {
        const popover = ensureStocktwitsSummaryPopover();
        popover.dataset.ticker = ticker || '';
        setStocktwitsPopoverState(state);
        popover.classList.add('visible');
        positionStocktwitsPopover(anchorEl);
    }

    function formatSummaryTimestamp(isoString) {
        if (!isoString) return '';
        const date = new Date(isoString);
        if (Number.isNaN(date.getTime())) return '';
        return date.toLocaleString(undefined, {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit'
        });
    }

    function fetchStocktwitsSummary(ticker) {
        const normalized = (ticker || '').toUpperCase();
        if (!normalized) {
            return Promise.reject(new Error('Invalid ticker.'));
        }
        if (stocktwitsSummaryCache.has(normalized)) {
            return Promise.resolve(stocktwitsSummaryCache.get(normalized));
        }
        if (stocktwitsSummaryPromises.has(normalized)) {
            return stocktwitsSummaryPromises.get(normalized);
        }
        const promise = fetch(`/stocktwits/${encodeURIComponent(normalized)}/summary`)
            .then(res => {
                if (!res.ok) {
                    throw new Error('Summary unavailable right now.');
                }
                return res.json();
            })
            .then(data => {
                stocktwitsSummaryCache.set(normalized, data);
                stocktwitsSummaryPromises.delete(normalized);
                return data;
            })
            .catch(err => {
                stocktwitsSummaryPromises.delete(normalized);
                throw err;
            });
        stocktwitsSummaryPromises.set(normalized, promise);
        return promise;
    }

    async function toggleStocktwitsSummaryPopover(anchorEl, ticker, name) {
        if (!anchorEl || !ticker) return;
        const normalized = ticker.toUpperCase();
        const headingTitle = name
            ? `${name} (${normalized}) on StockTwits`
            : `${normalized} on StockTwits`;
        const visiblePopover = stocktwitsSummaryPopover && stocktwitsSummaryPopover.classList.contains('visible');
        if (visiblePopover && stocktwitsSummaryPopover.dataset.ticker === normalized) {
            hideStocktwitsPopover();
            return;
        }
        showStocktwitsPopover(anchorEl, normalized, {
            heading: headingTitle,
            body: 'Loading summary…',
            meta: '',
            modifier: 'loading'
        });
        try {
            const data = await fetchStocktwitsSummary(normalized);
            const summaryText = (data.summary || '').trim() || 'No StockTwits summary available yet.';
            const timestampText = formatSummaryTimestamp(data.summary_meta && data.summary_meta.summary_at);
            showStocktwitsPopover(anchorEl, normalized, {
                heading: headingTitle,
                body: summaryText,
                meta: timestampText ? `Updated ${timestampText}` : ''
            });
        } catch (error) {
            const message = (error && error.message) || 'Summary unavailable.';
            showStocktwitsPopover(anchorEl, normalized, {
                heading: headingTitle,
                body: message,
                meta: '',
                modifier: 'error'
            });
        }
    }

    function renderTrendingData(source, items) {
        if (source === 'stocktwits') {
            renderStocktwitsList(items);
        } else if (source === 'reddit') {
            renderRedditList(items);
        } else if (source === 'volume') {
            renderVolumeList(items);
        }
    }

    function showWatchlistSection(show) {
        if (!watchlistSectionEl) return;
        if (isWatchlistPage) {
            watchlistSectionEl.style.display = show === false ? 'none' : '';
            return;
        }
        if (isTrendingPage || isSearchPage || isMarketPage) {
            watchlistSectionEl.style.display = 'none';
            return;
        }
        if (!watchlistHasEntries) {
            watchlistSectionEl.style.display = 'none';
            return;
        }
        watchlistSectionEl.style.display = show ? '' : 'none';
    }

    function setHomePrimaryPanel(panel) {
        if (!isHomePage) return;
        homePrimaryPanel = 'watchlist';
        const shouldShowWatchlist = watchlistHasEntries;
        showWatchlistSection(shouldShowWatchlist);
        showTrendingSection(false);
        if (shouldShowWatchlist) {
            refreshWatchlistPrices();
        }
    }

    function initializeHomePanel(watchlistCount) {
        if (!isHomePage || homePanelInitialized) return;
        homePanelInitialized = true;
        setHomePrimaryPanel('watchlist');
    }

    async function fetchQuote(symbol) {
        if (!symbol) return null;
        try {
            const res = await fetch(`/quote/${encodeURIComponent(symbol)}`);
            if (!res.ok) {
                throw new Error(`Quote fetch failed for ${symbol}`);
            }
            return await res.json();
        } catch (error) {
            console.error('Quote fetch error:', error);
            return null;
        }
    }

    async function fetchJson(url) {
        const response = await fetch(url);
        if (!response.ok) {
            let message = 'Request failed';
            try {
                const data = await response.json();
                if (data && data.error) {
                    message = data.error;
                }
            } catch (err) {
                // ignore parse errors
            }
            throw new Error(message);
        }
        return response.json();
    }

    async function refreshWatchlistPrices() {
        if (watchlistPriceRefreshPromise || !watchlistHasEntries) return;
        const items = readWatchlist();
        if (!items.length) return;
        watchlistPriceRefreshPromise = (async () => {
            try {
                const quotes = await Promise.all(items.map(item => fetchQuote(item.symbol)));
                let changed = false;
                const nextItems = items.map((item, idx) => {
                    const quote = quotes[idx];
                    if (quote && Number.isFinite(quote.price)) {
                        changed = true;
                        return {
                            ...item,
                            lastPrice: Number(quote.price),
                            lastUpdated: quote.timestamp || new Date().toISOString()
                        };
                    }
                    return item;
                });
                if (changed) {
                    writeWatchlist(nextItems);
                    renderWatchlist();
                }
            } catch (err) {
                console.error('Watchlist refresh error:', err);
            } finally {
                watchlistPriceRefreshPromise = null;
            }
        })();
        return watchlistPriceRefreshPromise;
    }

    function hideWatchlistForStockView() {
        if (!watchlistSectionEl) return;
        if (isWatchlistPage) {
            watchlistSectionEl.style.display = 'none';
        } else {
            showWatchlistSection(false);
        }
    }

    async function fetchSentimentWithTimeout(article, companyName, idx, timeoutMs, runId) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const normalizedRunId = runId === undefined || runId === null ? '' : String(runId);
            const res = await fetch('/sentiment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title: article.title,
                    description: article.description,
                    company_name: companyName,
                    article_id: idx,
                    sentiment_run_id: normalizedRunId || undefined
                }),
                signal: controller.signal
            });
            const payload = await res.json();
            if (!res.ok) {
                throw new Error(payload.error || 'Failed to analyze sentiment');
            }
            return payload.sentiment || 'neutral';
        } finally {
            clearTimeout(timeoutId);
        }
    }

    function renderStocktwitsList(items) {
        const list = trendingLists.stocktwits;
        if (!list) return;
        hideStocktwitsPopover();
        if (!Array.isArray(items) || items.length === 0) {
            showTrendingMessage('stocktwits', 'No StockTwits data available right now.');
            return;
        }

        list.innerHTML = '';
        items.slice(0, 10).forEach((item, idx) => {
            const div = document.createElement('div');
            div.className = 'ticker-row';
            div.dataset.ticker = (item.ticker || '').toUpperCase();
            div.onclick = () => {
                document.getElementById('companyInput').value = item.ticker;
                searchStock();
            };

            const changeValue = item.price_change_percent ?? item.change_percent;
            const priceValue = formatOptionalCurrency(item.price);
            const priceClass = Number(changeValue) < 0 ? 'down' : 'up';
            const safePriceText = escapeHtml(priceValue || '--');
            const trendingScore = Number(item.trending_score);
            const scoreLabel = Number.isFinite(trendingScore) ? trendingScore.toFixed(2) : '--';
            const safeTicker = escapeHtml(item.ticker || 'N/A');
            const safeName = escapeHtml(item.name || '');
            const safeRank = Number.isFinite(Number(item.rank)) ? Number(item.rank) : (idx + 1);
            const infoButtonHtml = item.ticker && (item.rank || idx + 1) <= 10
                ? '<button class="trending-info-btn" type="button" aria-label="View StockTwits summary"><span class="icon">i</span></button>'
                : '';

            div.innerHTML = `
                <span class="ticker-rank">${safeRank}</span>
                <div class="ticker-info">
                    <div class="ticker-symbol">${safeTicker} ${infoButtonHtml}</div>
                    <div class="ticker-name">${safeName}</div>
                    <span class="trending-change-inline" data-change-inline></span>
                </div>
                <div class="score-wrap">
                    <div class="score-val">${scoreLabel}</div>
                    <div class="score-label">Trend Score</div>
                </div>
                <div class="ticker-right">
                    <div class="ticker-price ${priceClass}">${safePriceText}</div>
                    <div data-change-right></div>
                </div>
            `;
            const infoBtn = div.querySelector('.trending-info-btn');
            if (infoBtn) {
                infoBtn.addEventListener('click', (event) => {
                    event.stopPropagation();
                    toggleStocktwitsSummaryPopover(infoBtn, item.ticker, item.name);
                });
            }
            updateTrendingChangeBadges(div, changeValue);
            list.appendChild(div);
        });
    }

    function renderRedditList(items) {
        const list = trendingLists.reddit;
        if (!list) return;
        if (!Array.isArray(items) || items.length === 0) {
            showTrendingMessage('reddit', 'No Reddit trending data available.');
            return;
        }

        list.innerHTML = '';
        items.slice(0, 10).forEach((item, idx) => {
            const div = document.createElement('div');
            div.className = 'ticker-row';
            div.dataset.ticker = (item.ticker || '').toUpperCase();
            div.onclick = () => {
                document.getElementById('companyInput').value = item.ticker;
                searchStock();
            };
            const pct = Number(item.pct_increase || 0);
            const pctClass = pct < 0 ? 'negative' : 'positive';
            const pctLabel = `${pct >= 0 ? '+' : ''}${Math.round(pct)}% mentions`;
            const safeTicker = escapeHtml(item.ticker || 'N/A');
            const safeName = escapeHtml(item.name || '');
            const safeRank = Number.isFinite(Number(item.rank)) ? Number(item.rank) : (idx + 1);
            const safePctLabel = escapeHtml(pctLabel);
            const mentionsLabel = Number.isFinite(Number(item.mentions)) ? `${formatNumber(Number(item.mentions))} mentions` : safePctLabel;
            const changeValue = item.price_change_percent ?? item.change_percent;
            const priceValue = formatOptionalCurrency(item.price);
            const priceClass = Number(changeValue) < 0 ? 'down' : 'up';
            const safePriceText = escapeHtml(priceValue || '--');

            div.innerHTML = `
                <span class="ticker-rank">${safeRank}</span>
                <div class="ticker-info">
                    <div class="ticker-symbol">${safeTicker}</div>
                    <div class="ticker-name">${safeName}</div>
                    <div class="ticker-mentions">${mentionsLabel}</div>
                    <div class="ticker-mentions-delta ${pctClass}">${safePctLabel}</div>
                    <span class="trending-change-inline" data-change-inline></span>
                </div>
                <div class="ticker-right">
                    <div class="ticker-price ${priceClass}">${safePriceText}</div>
                    <div data-change-right></div>
                </div>
            `;
            updateTrendingChangeBadges(div, changeValue);
            list.appendChild(div);
        });
    }

    function renderVolumeList(items) {
        const list = trendingLists.volume;
        if (!list) return;
        if (!Array.isArray(items) || items.length === 0) {
            showTrendingMessage('volume', 'No volume leaders available.');
            return;
        }

        list.innerHTML = '';
        items.slice(0, 10).forEach((item, idx) => {
            const div = document.createElement('div');
            div.className = 'ticker-row';
            div.dataset.ticker = (item.ticker || '').toUpperCase();
            div.onclick = () => {
                document.getElementById('companyInput').value = item.ticker;
                searchStock();
            };
            const volumeCount = Number(item.volume);
            const volumeLabel = Number.isFinite(volumeCount) && volumeCount > 0
                ? `${formatNumber(volumeCount)} shares`
                : 'Volume unavailable';
            const changeValue = item.price_change_percent ?? item.change_percent;
            const safeTicker = escapeHtml(item.ticker || 'N/A');
            const safeName = escapeHtml(item.name || '');
            const safeRank = Number.isFinite(Number(item.rank)) ? Number(item.rank) : (idx + 1);
            const priceValue = formatOptionalCurrency(item.price);
            const priceClass = Number(changeValue) < 0 ? 'down' : 'up';
            const safePriceText = escapeHtml(priceValue || '--');

            div.innerHTML = `
                <span class="ticker-rank">${safeRank}</span>
                <div class="ticker-info">
                    <div class="ticker-symbol">${safeTicker}</div>
                    <div class="ticker-name">${safeName}</div>
                    <div class="ticker-mentions">${volumeLabel}</div>
                    <span class="trending-change-inline" data-change-inline></span>
                </div>
                <div class="ticker-right">
                    <div class="ticker-price ${priceClass}">${safePriceText}</div>
                    <div data-change-right></div>
                </div>
            `;
            updateTrendingChangeBadges(div, changeValue);
            list.appendChild(div);
        });
    }

    function renderMarketSummaryIndices(indices) {
        if (!indices || !indices.length) return null;
        const wrapper = document.createElement('div');
        wrapper.className = 'market-summary-indices';
        indices.forEach(item => {
            const card = document.createElement('div');
            card.className = 'market-index-item';
            const label = document.createElement('div');
            label.className = 'market-index-label';
            label.textContent = item.label || item.symbol || 'Index';
            const price = document.createElement('div');
            price.className = 'market-index-price';
            price.textContent = Number.isFinite(item.close) ? formatCurrency(item.close) : '—';
            const change = document.createElement('div');
            change.className = 'market-index-change';
            if (Number.isFinite(item.change_percent)) {
                change.textContent = formatSignedPercentage(item.change_percent);
                change.classList.add(item.change_percent >= 0 ? 'positive' : 'negative');
            } else {
                change.textContent = '—';
            }
            card.appendChild(label);
            card.appendChild(price);
            card.appendChild(change);
            wrapper.appendChild(card);
        });
        return wrapper;
    }

    function renderMarketSummaryHeadlines(headlines) {
        if (!headlines || !headlines.length) return null;
        const container = document.createElement('div');
        container.className = 'market-summary-headlines';
        const title = document.createElement('h3');
        title.textContent = 'Headline drivers';
        container.appendChild(title);
        const list = document.createElement('ul');
        headlines.forEach(item => {
            const li = document.createElement('li');
            const linkTarget = item.link && item.link.startsWith('http') ? item.link : null;
            const headlineEl = linkTarget ? document.createElement('a') : document.createElement('span');
            headlineEl.textContent = item.title || 'Story';
            if (linkTarget) {
                headlineEl.href = linkTarget;
                headlineEl.target = '_blank';
                headlineEl.rel = 'noopener noreferrer';
            }
            const meta = document.createElement('span');
            const metaParts = [];
            if (item.source) metaParts.push(item.source);
            if (item.publishedAt) metaParts.push(formatDate(item.publishedAt));
            meta.textContent = metaParts.join(' • ');
            li.appendChild(headlineEl);
            if (metaParts.length) {
                li.appendChild(meta);
            }
            list.appendChild(li);
        });
        container.appendChild(list);
        return container;
    }

    function renderMarketSummaryHero(imageData, titleText) {
        if (!imageData || !imageData.url) return null;
        const hero = document.createElement('div');
        hero.className = 'market-summary-hero';
        const img = document.createElement('img');
        img.src = imageData.url;
        img.alt = imageData.description || `Market illustration for ${titleText || 'market summary'}`;
        img.loading = 'lazy';
        hero.appendChild(img);

        const photographerName = imageData.photographer || imageData.photographer_username || 'Unsplash photographer';
        const photographerProfile = imageData.photographer_profile || unsplashReferralFallback;
        const unsplashLink = imageData.unsplash_link || unsplashReferralFallback;
        const attribution = document.createElement('div');
        attribution.className = 'unsplash-attribution';
        attribution.innerHTML = `Photo by <a href="${photographerProfile}" target="_blank" rel="noopener noreferrer">${photographerName}</a> on <a href="${unsplashLink}" target="_blank" rel="noopener noreferrer">Unsplash</a>`;
        hero.appendChild(attribution);

        return hero;
    }

    function renderMarketSummaryLatest(article) {
        if (!marketSummaryLatestEl) return;
        marketSummaryLatestEl.innerHTML = '';
        if (!article) {
            marketSummaryLatestEl.innerHTML = '<div class="market-summary-empty">Latest edition is not available yet. Check back after 4:15 PM ET.</div>';
            return;
        }

        const buildSummaryChips = (indices) => {
            if (!Array.isArray(indices) || !indices.length) {
                return '<span class="chip chip-neutral">No index data</span>';
            }
            return indices.slice(0, 3).map((idx) => {
                const label = escapeHtml(idx.label || idx.symbol || 'Index');
                const raw = Number(idx.change_percent);
                if (Number.isFinite(raw)) {
                    const cls = raw > 0 ? 'chip-up' : (raw < 0 ? 'chip-down' : 'chip-neutral');
                    return `<span class="chip ${cls}">${label} ${formatSignedPercentage(raw)}</span>`;
                }
                return `<span class="chip chip-neutral">${label} —</span>`;
            }).join('');
        };

        const buildSummaryBody = (bodyText, expanded = false) => {
            const text = (bodyText || '').trim();
            if (!text) {
                return '<div class="summary-body">Summary coming soon.</div>';
            }
            return `<div class="summary-body${expanded ? ' expanded' : ''}">${escapeHtml(text)}</div>`;
        };

        if (!marketSummarySlug) {
            const meta = buildMarketSummaryDateMeta(article, {
                isLatest: true,
                referenceDate: new Date()
            });
            const headline = escapeHtml(article.title || 'Market Summary');
            const card = document.createElement('article');
            card.className = 'summary-card latest';
            card.innerHTML = `
                <div class="summary-card-top">
                    <div class="day-col">
                        <div class="day-context">${escapeHtml(meta.context)}</div>
                        <div class="day-name">${escapeHtml(meta.dayName)} <span class="latest-badge">LATEST</span></div>
                        <div class="day-date">${escapeHtml(meta.dayDate)}</div>
                    </div>
                    <div class="divider-v"></div>
                    <div class="summary-content">
                        <div class="summary-headline">${headline}</div>
                        ${buildSummaryBody(article.body, true)}
                    </div>
                </div>
                <div class="summary-card-footer">
                    <div class="index-chips">${buildSummaryChips(article.indices || [])}</div>
                </div>
            `;
            if (article.permalink) {
                card.addEventListener('click', () => {
                    window.location.href = article.permalink;
                });
            }
            marketSummaryLatestEl.appendChild(card);
            return;
        }

        const meta = buildMarketSummaryDateMeta(article);
        const card = document.createElement('article');
        card.className = 'summary-card summary-card-detail';
        card.innerHTML = `
            <div class="summary-card-top summary-card-detail-top">
                <div class="day-col">
                    <div class="day-context">${escapeHtml(meta.context)}</div>
                    <div class="day-name">${escapeHtml(meta.dayName)}</div>
                    <div class="day-date">${escapeHtml(meta.dayDate)}</div>
                </div>
                <div class="divider-v"></div>
                <div class="summary-content">
                    <div class="summary-headline">${escapeHtml(article.title || 'Market Summary')}</div>
                    <div class="summary-detail-published">Published ${escapeHtml(formatMarketSummaryTimestamp(article.published_at || article.created_at))}</div>
                </div>
            </div>
        `;
        const heroEl = renderMarketSummaryHero(article.image, article.title);
        if (heroEl) card.appendChild(heroEl);

        const bodyEl = document.createElement('div');
        bodyEl.className = 'summary-detail-body';
        createParagraphNodes(bodyEl, article.body);
        card.appendChild(bodyEl);

        const indicesEl = renderMarketSummaryIndices(article.indices || []);
        if (indicesEl) {
            card.appendChild(indicesEl);
        }
        const headlinesEl = renderMarketSummaryHeadlines(article.headlines || []);
        if (headlinesEl) {
            card.appendChild(headlinesEl);
        }
        if (article.permalink && !marketSummarySlug) {
            const shareRow = document.createElement('div');
            shareRow.className = 'market-summary-share';
            const shareLink = document.createElement('a');
            shareLink.href = article.permalink;
            shareLink.textContent = 'Read Full Market Summary →';
            shareRow.appendChild(shareLink);
            card.appendChild(shareRow);
        }
        marketSummaryLatestEl.appendChild(card);
    }

    function renderMarketSummaryArchive(items, latestId) {
        if (!marketSummaryArchiveEl) return;
        marketSummaryArchiveEl.innerHTML = '';
        const filtered = (items || []).filter(item => {
            if (!latestId) return true;
            return item.id !== latestId;
        });
        if (!filtered.length) {
            marketSummaryArchiveEl.innerHTML = '<div class="market-summary-empty">No earlier editions</div>';
            return;
        }

        const buildSummaryChips = (indices) => {
            if (!Array.isArray(indices) || !indices.length) {
                return '<span class="chip chip-neutral">No index data</span>';
            }
            return indices.slice(0, 3).map((idx) => {
                const label = escapeHtml(idx.label || idx.symbol || 'Index');
                const raw = Number(idx.change_percent);
                if (Number.isFinite(raw)) {
                    const cls = raw > 0 ? 'chip-up' : (raw < 0 ? 'chip-down' : 'chip-neutral');
                    return `<span class="chip ${cls}">${label} ${formatSignedPercentage(raw)}</span>`;
                }
                return `<span class="chip chip-neutral">${label} —</span>`;
            }).join('');
        };

        const referenceDate = new Date();
        filtered.forEach((item) => {
            const href = item.permalink || (item.slug ? `/market-summary/${encodeURIComponent(item.slug)}` : '/market-summary');
            const meta = buildMarketSummaryDateMeta(item, { referenceDate });

            const card = document.createElement('a');
            card.className = 'summary-card';
            card.href = href;
            card.setAttribute('aria-label', `Read ${item.title || 'market summary'}`);
            card.innerHTML = `
                <div class="summary-card-top">
                    <div class="day-col">
                        <div class="day-context">${escapeHtml(meta.context)}</div>
                        <div class="day-name">${escapeHtml(meta.dayName)}</div>
                        <div class="day-date">${escapeHtml(meta.dayDate)}</div>
                    </div>
                    <div class="divider-v"></div>
                    <div class="summary-content">
                        <div class="summary-headline">${escapeHtml(item.title || 'Market Summary')}</div>
                        <div class="summary-body">${escapeHtml((item.body || '').trim() || 'Summary coming soon.')}</div>
                    </div>
                </div>
                <div class="summary-card-footer">
                    <div class="index-chips">${buildSummaryChips(item.indices || [])}</div>
                </div>
            `;
            marketSummaryArchiveEl.appendChild(card);
        });
    }

    function setActiveTrendingColumn(source) {
        const columns = document.querySelectorAll('.trending-column');
        columns.forEach(column => {
            column.classList.add('active');
        });
    }

    function setTrendingTabsActive(source) {
        const normalized = normalizeTrendingSource(source);
        currentTrendingSource = normalized;
        document.body.dataset.trendingSource = normalized;
        const tabs = document.querySelectorAll('.trending-tab-btn');
        tabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.source === normalized);
        });
        setActiveTrendingColumn(normalized);
    }

    function navigateToTrendingSource(source, { replaceState = false } = {}) {
        const normalized = normalizeTrendingSource(source);
        const targetPath = buildTrendingPath(normalized);
        if (isTrendingPage) {
            if (window.location.pathname !== targetPath) {
                const historyMethod = replaceState ? 'replaceState' : 'pushState';
                if (typeof history[historyMethod] === 'function') {
                    history[historyMethod]({ trendingSource: normalized }, '', targetPath);
                }
            }
            setTrendingTabsActive(normalized);
            return;
        }
        window.location.href = targetPath;
    }

    function initializeTrendingTabs() {
        const tabs = document.querySelectorAll('.trending-tab-btn');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const source = normalizeTrendingSource(tab.dataset.source);
                if (isTrendingPage) {
                    if (source === currentTrendingSource) return;
                    navigateToTrendingSource(source);
                    return;
                }
                setTrendingTabsActive(source);
            });
        });
        setTrendingTabsActive(currentTrendingSource);
    }

    async function searchStock(forcedQuery = null) {
        const inputEl = document.getElementById('companyInput');
        const companyName = (forcedQuery !== null ? forcedQuery : (inputEl ? inputEl.value : '')).trim();

        if (!companyName) {
            showError('Enter a company name');
            return;
        }

        addRecentSearch(companyName);
        hideSearchSuggestions();

        if (!isSearchPage) {
            window.location.href = `/results?q=${encodeURIComponent(companyName)}`;
            return;
        }

        if (watchlistToggleBtn) {
            updateWatchlistToggle(null);
        }

        showTrendingSection(false);
        activeMovementInsightRequestId += 1;
        renderMovementInsight(null);
        document.getElementById('loading').style.display = 'block';
        document.getElementById('results').style.display = 'none';
        document.getElementById('errorMessage').innerHTML = '';
        document.getElementById('searchBtn').disabled = true;

        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    company_name: companyName
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                const errorMessage = (data && data.error) ? data.error : 'Failed to fetch data';
                const error = new Error(errorMessage);
                error.status = response.status;
                throw error;
            }

            displayResults(data);
            currentSymbol = data.stock_info.symbol;
            historicalData = { '1d': data.historical_data };
            loadMovementInsight(data.stock_info);

            document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
            document.querySelector('.chart-tab[data-period="1d"]').classList.add('active');

            document.getElementById('results').style.display = 'block';
        } catch (error) {
            const consolePayload = {
                status: error && error.status,
                message: (error && error.message) || 'Unknown search error'
            };
            console.error('Search error:', consolePayload);
            showError('An error ocurred while processing your request', { includeRetry: true });
        } finally {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('searchBtn').disabled = false;
        }
    }

    function showError(message, options = {}) {
        const { includeRetry = false } = options;
        const container = document.getElementById('errorMessage');
        if (!container) return;
        container.innerHTML = '';
        const errorEl = document.createElement('div');
        errorEl.className = 'error-message';
        errorEl.textContent = message || 'An unexpected error occurred.';
        container.appendChild(errorEl);
        if (includeRetry) {
            const retryBtn = document.createElement('button');
            retryBtn.className = 'retry-button';
            retryBtn.type = 'button';
            retryBtn.textContent = 'Try again';
            retryBtn.onclick = () => window.location.reload();
            container.appendChild(retryBtn);
        }
    }

    function displayResults(data) {
        const stockInfo = data.stock_info;
        const sentimentRunId = ++activeSentimentRunId;
        resetSentimentSummaryUI();
        latestSentimentCounts = { positive: 0, negative: 0, neutral: 0 };
        setSentimentStatus('Analyzing article sentiment...');
        updateWatchlistToggle(stockInfo.symbol, stockInfo.companyName, stockInfo.price);
        updateWatchlistSnapshot(stockInfo.symbol, stockInfo.price, stockInfo.companyName);
        const articles = data.articles || [];
        renderMovementInsight(null);
        hideWatchlistForStockView();

        document.getElementById('companyName').textContent = stockInfo.companyName;
        const symbolEl = document.getElementById('companySymbol');
        if (symbolEl) {
            symbolEl.textContent = stockInfo.symbol || '--';
        }
        const logoEl = document.getElementById('stockLogo');
        if (logoEl) {
            logoEl.textContent = (stockInfo.symbol || '?').charAt(0);
        }
        document.getElementById('stockPrice').textContent = formatCurrency(stockInfo.price);

        const changeClass = stockInfo.change >= 0 ? 'positive' : 'negative';
        const changeSign = stockInfo.change >= 0 ? '+' : '';
        document.getElementById('priceChange').innerHTML =
            `<span class="${changeClass}">${changeSign}${formatCurrency(stockInfo.change)} (${changeSign}${formatPercentage(stockInfo.changePercent)}) Today</span>`;

        if (stockInfo.afterHoursChange !== null && stockInfo.afterHoursChange !== undefined) {
            const ahChangeClass = stockInfo.afterHoursChange >= 0 ? 'positive' : 'negative';
            const ahChangeSign = stockInfo.afterHoursChange >= 0 ? '+' : '';
            document.getElementById('afterHoursChange').innerHTML =
                `<span class="${ahChangeClass}">${ahChangeSign}${formatCurrency(stockInfo.afterHoursChange)} (${ahChangeSign}${formatPercentage(stockInfo.afterHoursChangePercent)}) After-hours</span>`;
        } else {
            document.getElementById('afterHoursChange').innerHTML = '';
        }

        if (data.historical_data && data.historical_data.length > 0) {
            createChart(data.historical_data);
        } else {
            createChart([]);
        }

        document.getElementById('aboutSymbol').textContent = stockInfo.symbol;
        const aboutTextElement = document.getElementById('aboutText');
        const toggleBtn = document.getElementById('toggleAboutBtn');
        const defaultDescription = `${stockInfo.companyName} (${stockInfo.symbol}) engages in various business operations. Stock data is updated in real-time.`;
        const fullDescription = (stockInfo.description || defaultDescription).trim();
        const shortText = getFirstSentence(fullDescription) || fullDescription;
        const needsToggle = shortText.length < fullDescription.length;

        aboutTextElement.dataset.fullText = fullDescription;
        aboutTextElement.dataset.shortText = shortText;
        aboutTextElement.textContent = aboutTextElement.dataset.shortText;
        aboutTextElement.classList.remove('expanded');

        if (needsToggle) {
            toggleBtn.style.display = 'inline';
            toggleBtn.classList.remove('is-hidden');
            toggleBtn.textContent = 'Read more ↓';
            toggleBtn.dataset.expanded = 'false';
            toggleBtn.onclick = () => {
                const expanded = toggleBtn.dataset.expanded === 'true';
                if (expanded) {
                    aboutTextElement.textContent = aboutTextElement.dataset.shortText;
                    aboutTextElement.classList.remove('expanded');
                    toggleBtn.textContent = 'Read more ↓';
                    toggleBtn.dataset.expanded = 'false';
                } else {
                    aboutTextElement.textContent = aboutTextElement.dataset.fullText;
                    aboutTextElement.classList.add('expanded');
                    toggleBtn.textContent = 'Read less ↑';
                    toggleBtn.dataset.expanded = 'true';
                }
            };
        } else {
            toggleBtn.style.display = 'none';
            toggleBtn.classList.add('is-hidden');
            toggleBtn.textContent = '';
            toggleBtn.dataset.expanded = 'false';
            toggleBtn.onclick = null;
            aboutTextElement.classList.remove('expanded');
        }
        document.getElementById('ceo').textContent = stockInfo.ceo || 'N/A';
        document.getElementById('employees').textContent = stockInfo.employees ? formatNumber(stockInfo.employees) : 'N/A';

        let location = [];
        if (stockInfo.city) location.push(stockInfo.city);
        if (stockInfo.state) location.push(stockInfo.state);
        if (stockInfo.country && !stockInfo.state) location.push(stockInfo.country);
        document.getElementById('headquarters').textContent = location.join(', ') || 'N/A';

        document.getElementById('founded').textContent = stockInfo.yearFounded;

        document.getElementById('marketCap').textContent = stockInfo.marketCap ? '$' + formatNumber(stockInfo.marketCap) : 'N/A';
        document.getElementById('peRatio').textContent = stockInfo.peRatio ? stockInfo.peRatio.toFixed(2) : 'N/A';
        document.getElementById('dividendYield').textContent = stockInfo.dividendYield ? formatPercentage(stockInfo.dividendYield) : 'N/A';
        document.getElementById('avgVolume').textContent = stockInfo.avgVolume ? formatNumber(stockInfo.avgVolume) : 'N/A';
        document.getElementById('highToday').textContent = stockInfo.dayHigh ? formatCurrency(stockInfo.dayHigh) : 'N/A';
        document.getElementById('lowToday').textContent = stockInfo.dayLow ? formatCurrency(stockInfo.dayLow) : 'N/A';
        document.getElementById('openPrice').textContent = stockInfo.openPrice ? formatCurrency(stockInfo.openPrice) : 'N/A';
        document.getElementById('volume').textContent = stockInfo.volume ? formatNumber(stockInfo.volume) : 'N/A';
        document.getElementById('fiftyTwoWeekHigh').textContent = stockInfo.fiftyTwoWeekHigh ? formatCurrency(stockInfo.fiftyTwoWeekHigh) : 'N/A';
        document.getElementById('fiftyTwoWeekLow').textContent = stockInfo.fiftyTwoWeekLow ? formatCurrency(stockInfo.fiftyTwoWeekLow) : 'N/A';
        renderSentimentSummaryUI(latestSentimentCounts, 0, true);

        const newsContainer = document.getElementById('newsArticles');
        newsContainer.innerHTML = '';

        if (articles.length === 0) {
            newsContainer.innerHTML = '<div class="news-pending">No news articles available.</div>';
            renderSentimentSummaryUI(latestSentimentCounts, 0, false);
            setSentimentStatus('No articles to analyze.');
            return;
        }

        const pendingNotice = document.createElement('div');
        pendingNotice.className = 'news-pending';
        pendingNotice.textContent = 'Analyzing sentiment so we can highlight the most relevant articles...';
        newsContainer.appendChild(pendingNotice);

        progressivelyAnalyzeSentiment(
            articles,
            stockInfo.companyName || stockInfo.symbol,
            sentimentRunId,
            newsContainer,
            pendingNotice
        );
    }

    function renderMovementInsightLoading(changePercent) {
        if (!movementBox || !movementSummaryEl || !movementMetaEl || !movementSourcesEl) {
            return;
        }

        movementBox.style.display = 'block';
        movementSummaryEl.textContent = 'Loading what\'s happening...';

        const normalizedChangePercent = Number(changePercent);
        movementMetaEl.textContent = Number.isFinite(normalizedChangePercent)
            ? `Moved ${formatSignedPercentage(normalizedChangePercent)} today`
            : 'Intraday move > 3%';

        movementSourcesEl.innerHTML = '';
        movementSourcesEl.style.display = 'none';
    }

    async function loadMovementInsight(stockInfo) {
        const changePercent = Number(stockInfo && stockInfo.changePercent);
        if (!Number.isFinite(changePercent) || Math.abs(changePercent) < 3) {
            renderMovementInsight(null);
            return;
        }

        const requestId = ++activeMovementInsightRequestId;
        renderMovementInsightLoading(changePercent);

        try {
            const response = await fetch('/movement-insight', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ stock_info: stockInfo })
            });

            const payload = await response.json();
            if (requestId !== activeMovementInsightRequestId) {
                return;
            }

            if (!response.ok) {
                throw new Error((payload && payload.error) || 'Failed to load movement insight');
            }

            renderMovementInsight(payload.movement_insight || null);
        } catch (error) {
            if (requestId !== activeMovementInsightRequestId) {
                return;
            }
            console.warn('Movement insight load error:', error);
            renderMovementInsight(null);
        }
    }

    async function progressivelyAnalyzeSentiment(articles, companyName, runId, targetContainer, pendingElement) {
        if (!articles || !articles.length) {
            renderSentimentSummaryUI(latestSentimentCounts, 0, false);
            setSentimentStatus('No articles to analyze.');
            sentimentBatchSize = 0;
            updateSentimentProgress(0, false);
            return;
        }

        const newsContainer = targetContainer || document.getElementById('newsArticles');
        let waitingNotice = pendingElement || null;

        const total = articles.length;
        let completed = 0;
        sentimentBatchSize = total;
        updateSentimentProgress(0, true);
        setSentimentStatus(`Analyzing ${total} article${total === 1 ? '' : 's'}...`);
        const runIdentifier = String(runId);

        for (let idx = 0; idx < articles.length; idx++) {
            if (runId !== activeSentimentRunId) return;

            const article = articles[idx];
            let sentiment = null;
            let didTimeout = false;
            try {
                sentiment = await fetchSentimentWithTimeout(
                    article,
                    companyName,
                    idx,
                    SENTIMENT_REQUEST_TIMEOUT_MS,
                    runIdentifier
                );
            } catch (error) {
                didTimeout = error.name === 'AbortError';
                if (didTimeout) {
                    setSentimentStatus('Sentiment is taking longer than usual. Retrying...', 'error');
                    try {
                        sentiment = await fetchSentimentWithTimeout(
                            article,
                            companyName,
                            idx,
                            SENTIMENT_RETRY_TIMEOUT_MS,
                            runIdentifier
                        );
                        didTimeout = false;
                    } catch (retryError) {
                        didTimeout = retryError.name === 'AbortError';
                        console.error('Sentiment retry error:', retryError);
                    }
                } else {
                    console.error('Sentiment analysis error:', error);
                }
            }

            if (runId !== activeSentimentRunId) return;

            if (didTimeout && !sentiment) {
                setSentimentStatus('Sentiment request timed out. Some articles may be missing sentiment.', 'error');
            }

            const normalized = (sentiment || '').toString().trim().toLowerCase();
            if (['positive', 'negative', 'neutral'].includes(normalized)) {
                latestSentimentCounts[normalized] = (latestSentimentCounts[normalized] || 0) + 1;
            }

            if (waitingNotice && newsContainer && newsContainer.contains(waitingNotice)) {
                newsContainer.removeChild(waitingNotice);
                waitingNotice = null;
            }

            if (newsContainer && runId === activeSentimentRunId) {
                const { badgeText, badgeClass } = determineSentimentBadge(normalized, didTimeout, Boolean(sentiment));
                const articleElement = buildNewsArticleElement(
                    articles[idx],
                    badgeText,
                    badgeClass
                );
                newsContainer.appendChild(articleElement);
            }

            completed += 1;
            renderSentimentSummaryUI(latestSentimentCounts, completed, completed < total);
            setSentimentStatus(completed < total ? `Analyzed ${completed}/${total} articles...` : '');
        }
    }

    function determineSentimentBadge(normalizedSentiment, timedOut, hasRawValue) {
        if (['positive', 'negative', 'neutral'].includes(normalizedSentiment)) {
            return {
                badgeText: normalizedSentiment,
                badgeClass: `sentiment-badge ${normalizedSentiment}`
            };
        }

        if (timedOut) {
            return {
                badgeText: 'Timeout',
                badgeClass: 'sentiment-badge neutral muted'
            };
        }

        if (hasRawValue) {
            return {
                badgeText: 'Unclear',
                badgeClass: 'sentiment-badge neutral muted'
            };
        }

        return {
            badgeText: 'No result',
            badgeClass: 'sentiment-badge neutral muted'
        };
    }

    function sanitizeNewsText(value, fallback = '') {
        const raw = (value || '').toString().trim();
        if (!raw) {
            return fallback;
        }

        const entityDecoder = document.createElement('textarea');
        entityDecoder.innerHTML = raw;
        const decoded = (entityDecoder.value || raw).trim();

        const parsed = new DOMParser().parseFromString(decoded, 'text/html');
        const readable = (parsed.body?.textContent || decoded)
            .replace(/\u00a0/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();

        return readable || fallback;
    }

    function decodeHtmlEntities(value) {
        const entityDecoder = document.createElement('textarea');
        entityDecoder.innerHTML = (value || '').toString();
        return entityDecoder.value || (value || '').toString();
    }

    function buildNewsDescriptionContent(article) {
        const descriptionWrapper = document.createElement('span');
        const rawDescription = (article.description || '').toString().trim();

        if (!rawDescription) {
            descriptionWrapper.textContent = 'No summary provided.';
            return descriptionWrapper;
        }

        const decoded = decodeHtmlEntities(rawDescription).trim();
        const parsed = new DOMParser().parseFromString(decoded, 'text/html');
        const embeddedLink = parsed.body.querySelector('a[href]');

        const linkEl = document.createElement('a');
        const linkHref = embeddedLink?.getAttribute('href') || article.link || '';
        const linkText = embeddedLink
            ? sanitizeNewsText(embeddedLink.textContent, 'Read article')
            : sanitizeNewsText(decoded, 'No summary provided.');

        if (linkHref) {
            linkEl.href = linkHref;
            linkEl.target = '_blank';
            linkEl.rel = 'noopener noreferrer';
            linkEl.textContent = linkText;
            linkEl.addEventListener('click', event => event.stopPropagation());
            descriptionWrapper.appendChild(linkEl);
        } else {
            descriptionWrapper.textContent = linkText;
        }

        const embeddedSource = sanitizeNewsText(parsed.body.querySelector('font')?.textContent || '', '');
        const sourceText = embeddedSource || sanitizeNewsText(article.source, '');
        if (sourceText && descriptionWrapper.childNodes.length) {
            descriptionWrapper.appendChild(document.createTextNode('  '));
            const sourceEl = document.createElement('span');
            sourceEl.className = 'news-description-source';
            sourceEl.textContent = sourceText;
            descriptionWrapper.appendChild(sourceEl);
        }

        return descriptionWrapper;
    }

    function buildNewsArticleElement(article, badgeText, badgeClass) {
        const articleDiv = document.createElement('div');
        articleDiv.className = 'news-article news-item';

        if (article.link) {
            articleDiv.addEventListener('click', () => window.open(article.link, '_blank'));
        } else {
            articleDiv.style.cursor = 'default';
        }

        const publishedDate = article.publishedAt ? formatDate(article.publishedAt) : 'Recently updated';
        const sourceLabel = sanitizeNewsText(article.source, 'News');
        let tagClass = 'tag-neu';
        if ((badgeClass || '').includes('positive')) {
            tagClass = 'tag-pos';
        } else if ((badgeClass || '').includes('negative')) {
            tagClass = 'tag-neg';
        }

        const sentimentTagEl = document.createElement('span');
        sentimentTagEl.className = `news-sent-tag ${tagClass}`;
        sentimentTagEl.textContent = (badgeText || 'N/A').slice(0, 3);

        const bodyEl = document.createElement('div');
        bodyEl.className = 'news-body';

        const titleEl = document.createElement('div');
        titleEl.className = 'news-title news-title-text';
        titleEl.textContent = sanitizeNewsText(article.title, 'Untitled coverage');

        const metaEl = document.createElement('div');
        metaEl.className = 'news-meta';

        const sourceEl = document.createElement('span');
        sourceEl.className = 'news-source';
        sourceEl.textContent = sourceLabel;

        const timeEl = document.createElement('span');
        timeEl.className = 'news-time';
        timeEl.textContent = publishedDate;

        metaEl.appendChild(sourceEl);
        metaEl.appendChild(timeEl);

        bodyEl.appendChild(titleEl);
        bodyEl.appendChild(metaEl);

        articleDiv.appendChild(sentimentTagEl);
        articleDiv.appendChild(bodyEl);

        return articleDiv;
    }

    function renderMovementInsight(insight) {
        if (!movementBox || !movementSummaryEl || !movementMetaEl || !movementSourcesEl) {
            return;
        }

        if (!insight || !insight.summary) {
            movementBox.style.display = 'none';
            movementSummaryEl.textContent = '';
            movementMetaEl.textContent = '';
            movementSourcesEl.innerHTML = '';
            return;
        }

        movementBox.style.display = 'block';
        const changePercent = Number(insight.changePercent);
        movementMetaEl.textContent = Number.isFinite(changePercent)
            ? `Moved ${formatSignedPercentage(changePercent)} today`
            : 'Intraday move > 3%';
        movementSummaryEl.textContent = insight.summary;

        movementSourcesEl.innerHTML = '';
        const sources = Array.isArray(insight.sources) ? insight.sources : [];
        sources.forEach(source => {
            const li = document.createElement('li');
            const headline = source.headline || 'View source';
            const metaBits = [];
            if (source.source) metaBits.push(source.source);
            if (source.publishedAt) metaBits.push(formatDate(source.publishedAt));

            if (source.url) {
                const link = document.createElement('a');
                link.href = source.url;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                link.textContent = headline;
                li.appendChild(link);
            } else {
                li.textContent = headline;
            }

            if (metaBits.length) {
                const metaSpan = document.createElement('span');
                metaSpan.style.color = '#a1a1aa';
                metaSpan.style.fontSize = '12px';
                metaSpan.style.marginLeft = '6px';
                metaSpan.textContent = `• ${metaBits.join(' • ')}`;
                li.appendChild(metaSpan);
            }

            movementSourcesEl.appendChild(li);
        });

        movementSourcesEl.style.display = movementSourcesEl.children.length ? 'flex' : 'none';
    }

    function createChart(data) {
        const ctx = document.getElementById('stockChart').getContext('2d');

        if (chart) {
            chart.destroy();
        }

        let chartData, labels;

        if (data && data.length > 0) {
            chartData = data.map(point => point.price);
            labels = data.map(point => '');
        } else {
            chartData = [];
            labels = [];
        }

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    data: chartData,
                    borderColor: '#9333ea',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: {
                        left: 0,
                        right: 0,
                        top: 0,
                        bottom: 10
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return '$' + context.parsed.y.toFixed(2);
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        display: false,
                        grid: {
                            color: '#333',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#888',
                            padding: 10,
                            callback: function(value) {
                                return '$' + value.toFixed(0);
                            }
                        },
                        afterFit: (axis) => {
                            axis.width += 10;
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    document.addEventListener('DOMContentLoaded', function() {
        initializeHomeScrollEffect();
        initializeSearchSuggestions();
        const tabs = document.querySelectorAll('.chart-tab');

        tabs.forEach(tab => {
            tab.addEventListener('click', async function() {
                if (!currentSymbol) return;

                tabs.forEach(t => t.classList.remove('active'));
                this.classList.add('active');

                const period = this.getAttribute('data-period');

                if (historicalData[period]) {
                    createChart(historicalData[period]);
                } else {
                    try {
                        const response = await fetch(`/historical/${currentSymbol}/${period}`);
                        const result = await response.json();

                        if (result.data) {
                            historicalData[period] = result.data;
                            createChart(result.data);
                        }
                    } catch (error) {
                        console.error('Error fetching historical data:', error);
                    }
                }
            });
        });

        initializeTrendingTabs();
        const watchlistCount = renderWatchlist();
        initializeHomePanel(watchlistCount);

        if ((isHomePage || isWatchlistPage) && watchlistCount > 0) {
            refreshWatchlistPrices();
        }

        if (isMarketPage) {
            showWatchlistSection(false);
            showTrendingSection(false);
            loadMarketSummaryContent();
        } else if (isTrendingPage) {
            showTrendingSection(true);
            initializeTrendingLazyLoading({
                eagerSource: currentTrendingSource,
                includePrices: TRENDING_INCLUDE_PRICES,
                idlePrefetch: true
            });
            const canonicalPath = buildTrendingPath(currentTrendingSource);
            if (window.location.pathname !== canonicalPath) {
                navigateToTrendingSource(currentTrendingSource, { replaceState: true });
            }
        } else {
            const shouldShowTrending = isHomePage && homePrimaryPanel === 'trending';
            showTrendingSection(shouldShowTrending);
        }

        if (isSearchPage && initialSearchQuery) {
            if (companyInputEl) {
                companyInputEl.value = initialSearchQuery;
            }
            searchStock(initialSearchQuery);
        }

        if (watchlistToggleBtn) {
            watchlistToggleBtn.addEventListener('click', handleWatchlistToggleClick);
        }

        if (marketSummaryFormEl) {
            marketSummaryFormEl.addEventListener('submit', handleMarketSummarySignup);
        }
    });

    function formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diff = Math.floor((now - date) / 1000 / 60 / 60);

        if (diff < 24) {
            return diff + 'h';
        } else {
            const days = Math.floor(diff / 24);
            return days + 'd';
        }
    }

    if (companyInputEl) {
        companyInputEl.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchStock();
            }
        });
    }

    async function loadTrendingSource(source, options = {}) {
        const {
            forceRefresh = false,
            includePrices = TRENDING_INCLUDE_PRICES
        } = options;
        const cached = getFreshTrendingCache(source, includePrices);
        if (!forceRefresh && cached) {
            renderTrendingData(source, cached.data);
            if (!cached.includePrices) {
                hydrateTrendingPrices(source, cached.data);
            }
            return cached.data;
        }

        if (trendingFetchPromises[source]) {
            return trendingFetchPromises[source];
        }

        setTrendingListLoading(source);
        const fetchPromise = (async () => {
            try {
                let url = `/trending/${source}`;
                if (!includePrices) {
                    url += url.includes('?') ? '&' : '?';
                    url += 'include_prices=0';
                }
                const res = await fetch(url);
                if (!res.ok) {
                    throw new Error(`Failed to load ${source} data.`);
                }
                const data = await res.json();
                const items = data.data || [];
                setTrendingCache(source, items, includePrices);
                renderTrendingData(source, items);
                loadedTrendingSources.add(source);
                if (!includePrices) {
                    hydrateTrendingPrices(source, items);
                }
                return items;
            } catch (e) {
                console.error(`Trending ${source} error:`, e);
                const fallback = trendingCache[source];
                if (fallback && Array.isArray(fallback.data) && fallback.data.length) {
                    renderTrendingData(source, fallback.data);
                    if (!fallback.includePrices) {
                        hydrateTrendingPrices(source, fallback.data);
                    }
                } else {
                    showTrendingMessage(source, 'Failed to load data.', true);
                }
                return null;
            } finally {
                trendingFetchPromises[source] = null;
            }
        })();

        trendingFetchPromises[source] = fetchPromise;
        return fetchPromise;
    }

    function loadTrendingSourcesAsync(sources, options = {}) {
        const sourceList = (Array.isArray(sources) ? sources : TRENDING_SOURCES)
            .map(source => normalizeTrendingSource(source));
        const uniqueSources = Array.from(new Set(sourceList));
        return Promise.allSettled(uniqueSources.map(source => loadTrendingSource(source, options)));
    }

    function disconnectTrendingLazyObserver() {
        if (!trendingLazyObserver) return;
        trendingLazyObserver.disconnect();
        trendingLazyObserver = null;
    }

    function initializeTrendingLazyLoading(options = {}) {
        const {
            eagerSource = currentTrendingSource,
            includePrices = TRENDING_INCLUDE_PRICES,
            idlePrefetch = true
        } = options;

        if (trendingLazyInitialized) {
            return;
        }
        trendingLazyInitialized = true;

        const columns = Array.from(document.querySelectorAll('#trendingColumns .trending-column[data-source]'));
        if (!columns.length) {
            loadTrendingStocks({ includePrices });
            return;
        }

        const loadSourceIfNeeded = (source) => {
            const normalized = normalizeTrendingSource(source);
            if (loadedTrendingSources.has(normalized)) {
                return;
            }
            loadTrendingSource(normalized, { includePrices });
        };

        const prioritizedSource = normalizeTrendingSource(eagerSource);
        loadSourceIfNeeded(prioritizedSource);

        if (typeof window !== 'undefined' && typeof window.IntersectionObserver === 'function') {
            trendingLazyObserver = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) {
                        return;
                    }
                    const source = entry.target && entry.target.dataset ? entry.target.dataset.source : null;
                    if (!source) {
                        return;
                    }
                    loadSourceIfNeeded(source);
                    if (trendingLazyObserver) {
                        trendingLazyObserver.unobserve(entry.target);
                    }
                });
            }, {
                rootMargin: '220px 0px',
                threshold: 0.01
            });

            columns.forEach((column) => {
                const source = column.dataset ? column.dataset.source : null;
                if (!source || normalizeTrendingSource(source) === prioritizedSource) {
                    return;
                }
                trendingLazyObserver.observe(column);
            });
        } else {
            loadTrendingSourcesAsync(
                columns
                    .map(column => column.dataset && column.dataset.source)
                    .filter(source => source && normalizeTrendingSource(source) !== prioritizedSource),
                { includePrices }
            );
        }

        if (idlePrefetch) {
            queueIdleWork(() => {
                const remaining = columns
                    .map(column => column.dataset && column.dataset.source)
                    .filter(Boolean)
                    .filter(source => !loadedTrendingSources.has(normalizeTrendingSource(source)));
                if (remaining.length) {
                    loadTrendingSourcesAsync(remaining, { includePrices });
                }
            }, 900);
        }
    }

    function loadTrendingStocks(options = {}) {
        const mergedOptions = {
            includePrices: TRENDING_INCLUDE_PRICES,
            ...options
        };
        loadTrendingSourcesAsync(TRENDING_SOURCES, mergedOptions);
    }

    function ensureHomeTrendingLoaded() {
        if (homeTrendingInitialized || isTrendingPage) return;
        initializeTrendingLazyLoading({
            eagerSource: 'stocktwits',
            includePrices: TRENDING_INCLUDE_PRICES,
            idlePrefetch: true
        });
        homeTrendingInitialized = true;
    }

    async function loadMarketSummaryLatest() {
        setMarketSummaryLatestLoading();

        if (marketSummarySlug) {
            let standaloneArticle = initialMarketSummaryArticle;
            try {
                const articlePayload = await fetchJson(`/api/market-summary/${encodeURIComponent(marketSummarySlug)}`);
                if (articlePayload && articlePayload.article) {
                    standaloneArticle = articlePayload.article;
                }
            } catch (error) {
                if (!standaloneArticle) {
                    throw error;
                }
            }

            if (!standaloneArticle) {
                throw new Error('Market summary article unavailable.');
            }
            marketSummaryLatestArticleId = standaloneArticle.id || null;
            renderMarketSummaryLatest(standaloneArticle);
            return;
        }

        const latestPayload = await fetchJson('/api/market-summary/latest');
        const latestArticle = latestPayload && latestPayload.article;
        marketSummaryLatestArticleId = (latestArticle && latestArticle.id) || null;
        renderMarketSummaryLatest(latestArticle || null);
        syncMarketSummaryArchive();
    }

    async function loadMarketSummaryWeekGlance() {
        setMarketSummaryStatsLoading();
        try {
            const payload = await fetchJson('/api/market-summary/week-glance');
            const weekGlance = (payload && payload.week_glance) || [];
            renderMarketSummaryStats(weekGlance);
        } catch (error) {
            console.error('Week at a glance load error:', error);
            renderMarketSummaryStats([]);
        }
    }

    async function loadMarketSummaryArchive() {
        if (marketSummarySlug) {
            if (marketSummaryArchiveHeadingEl) {
                marketSummaryArchiveHeadingEl.style.display = 'none';
            }
            if (marketSummaryArchiveEl) {
                marketSummaryArchiveEl.innerHTML = '';
                marketSummaryArchiveEl.style.display = 'none';
            }
            return;
        }

        if (marketSummaryArchiveHeadingEl) {
            marketSummaryArchiveHeadingEl.style.display = '';
        }
        setMarketSummaryArchiveLoading();

        const archivePayload = await fetchJson('/api/market-summary/archive?limit=12');
        marketSummaryArchiveItems = (archivePayload && archivePayload.articles) || [];
        syncMarketSummaryArchive();
    }

    function loadMarketSummaryContent() {
        if (!marketSummarySectionEl) return;

        marketSummaryLatestArticleId = null;
        marketSummaryArchiveItems = [];

        const tasks = [
            loadMarketSummaryLatest().catch((error) => {
                console.error('Latest market summary load error:', error);
                renderMarketSummaryLatestError((error && error.message) || 'Unable to load market summary.');
            }),
            loadMarketSummaryWeekGlance(),
            loadMarketSummaryArchive().catch((error) => {
                console.error('Market summary archive load error:', error);
                renderMarketSummaryArchiveError((error && error.message) || 'Unable to load past editions.');
            })
        ];

        Promise.allSettled(tasks);
    }
    // Hide trending-section after a search, show on home
    function showTrendingSection(show) {
        const trendingSection = document.getElementById('trendingSection');
        if (!trendingSection) return;
        if (isTrendingPage) {
            trendingSection.style.display = '';
            return;
        }
        if (isWatchlistPage || isSearchPage || isMarketPage) {
            trendingSection.style.display = 'none';
            return;
        }
        if (show) {
            trendingSection.style.display = '';
            if (isHomePage) {
                ensureHomeTrendingLoaded();
            }
        } else {
            trendingSection.style.display = 'none';
        }
    }
    function openTrendingPage() {
        navigateToTrendingSource('stocktwits');
    }

    if (homeTitleEl) {
        homeTitleEl.addEventListener('click', (event) => {
            event.preventDefault();
            window.location.href = '/';
        });
    }

    if (trendingBtnEl) {
        trendingBtnEl.addEventListener('click', (event) => {
            if (!isTrendingPage) {
                return;
            }
            event.preventDefault();
            openTrendingPage();
        });
    }

    if (watchlistNavBtnEl) {
        watchlistNavBtnEl.addEventListener('click', (event) => {
            if (!isWatchlistPage) {
                return;
            }
            event.preventDefault();
            if (watchlistSectionEl) {
                watchlistSectionEl.style.display = '';
            }
            const count = renderWatchlist();
            if (count > 0) {
                refreshWatchlistPrices();
            }
        });
    }

    if (marketSummaryBtnEl) {
        marketSummaryBtnEl.addEventListener('click', (event) => {
            if (isMarketPage && !marketSummarySlug) {
                event.preventDefault();
                if (marketSummaryBackBtnEl) {
                    marketSummaryBackBtnEl.classList.add('is-hidden');
                }
            }
        });
    }

    document.addEventListener('click', (event) => {
        if (!stocktwitsSummaryPopover || !stocktwitsSummaryPopover.classList.contains('visible')) {
            return;
        }
        if (stocktwitsSummaryPopover.contains(event.target)) {
            return;
        }
        if (event.target.closest && event.target.closest('.trending-info-btn')) {
            return;
        }
        hideStocktwitsPopover();
    }, true);

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            hideStocktwitsPopover();
        }
    });

    window.addEventListener('scroll', () => {
        if (stocktwitsSummaryPopover && stocktwitsSummaryPopover.classList.contains('visible')) {
            hideStocktwitsPopover();
        }
    }, true);

    window.addEventListener('resize', () => {
        if (stocktwitsSummaryPopover && stocktwitsSummaryPopover.classList.contains('visible')) {
            hideStocktwitsPopover();
        }
    });

    if (isTrendingPage) {
        window.addEventListener('popstate', () => {
            const restoredSource = resolveTrendingSourceFromPath(window.location.pathname);
            setTrendingTabsActive(restoredSource);
        });
    }

    window.addEventListener('beforeunload', () => {
        disconnectTrendingLazyObserver();
    });
