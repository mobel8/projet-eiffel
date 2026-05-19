/**
 * EiffelPulse — Frontend
 * Single Page App Alpine.js + Chart.js
 */

const API_BASE = window.location.origin;

const ICON = {
  dashboard: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/></svg>`,
  forecast: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>`,
  reviews: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/></svg>`,
  chat: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>`,
  pricing: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"/></svg>`,
  about: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`,
};

// Couleurs Chart.js
const PALETTE = {
  gold: '#d4af37',
  goldLight: '#f4ecc8',
  goldDark: '#a87324',
  emerald: '#34d399',
  rose: '#fb7185',
  amber: '#fbbf24',
  blue: '#60a5fa',
  purple: '#a78bfa',
  midnight: '#1b2845',
  text: '#c6d2e3',
  grid: 'rgba(154, 175, 203, 0.08)',
};

Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.color = PALETTE.text;
Chart.defaults.borderColor = PALETTE.grid;

// Helpers
function fmtNum(n) {
  if (n === null || n === undefined) return '0';
  return new Intl.NumberFormat('fr-FR').format(n);
}
function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

function eiffelApp() {
  return {
    // State
    route: 'dashboard',
    now: '',
    nav: [
      { id: 'dashboard', label: 'Vue d\'ensemble', icon: ICON.dashboard },
      { id: 'forecast', label: 'Prédiction affluence', icon: ICON.forecast },
      { id: 'reviews', label: 'Analyse avis', icon: ICON.reviews },
      { id: 'chat', label: 'Assistant visiteur', icon: ICON.chat },
      { id: 'pricing', label: 'Veille tarifaire', icon: ICON.pricing },
      { id: 'about', label: 'À propos', icon: ICON.about },
    ],
    dashboard: null,
    forecast: null,
    forecastDate: todayStr(),
    historical: null,
    sentimentInput: '',
    sentimentResult: null,
    sentimentLoading: false,
    sampleReviews: [
      'La vue depuis le sommet est absolument magnifique, je recommande !',
      'Trop cher et file d\'attente énorme, déception totale.',
      'The elevator was broken and we had to climb, awful experience.',
      'Personal staff was very friendly and helpful, perfect visit.',
    ],
    latestReviews: [],
    filterSentiment: null,
    chatInput: '',
    chatLoading: false,
    messages: [],
    suggestedQuestions: [
      'Quels sont les horaires d\'ouverture ?',
      'Combien coûte un billet adulte ?',
      'La tour est-elle accessible aux PMR ?',
      'Y a-t-il un restaurant au sommet ?',
      'Quelle est la meilleure heure pour visiter ?',
      'How tall is the Eiffel Tower?',
      'Can I climb by the stairs?',
      'Combien de temps pour visiter ?',
    ],
    pricing: null,
    apiEndpoints: [
      { method: 'GET', path: '/api/health' },
      { method: 'POST', path: '/api/sentiment/predict' },
      { method: 'GET', path: '/api/sentiment/metrics' },
      { method: 'GET', path: '/api/forecast/day' },
      { method: 'GET', path: '/api/forecast/week' },
      { method: 'GET', path: '/api/forecast/historical' },
      { method: 'POST', path: '/api/chat' },
      { method: 'GET', path: '/api/chat/faq' },
      { method: 'GET', path: '/api/dashboard/reviews' },
      { method: 'GET', path: '/api/dashboard/competitors' },
    ],
    charts: {},

    async init() {
      this.messages = [this.welcomeMessage()];
      this.updateClock();
      setInterval(() => this.updateClock(), 60000);

      // Hash routing
      const hash = window.location.hash.replace('#', '');
      if (hash && this.nav.find(n => n.id === hash)) this.route = hash;
      window.addEventListener('hashchange', () => {
        const h = window.location.hash.replace('#', '');
        if (h && this.nav.find(n => n.id === h)) this.route = h;
      });

      await Promise.all([
        this.loadDashboard(),
        this.loadHistorical(),
        this.loadLatestReviews(),
        this.loadPricing(),
      ]);

      this.renderDashboardCharts();
      this.renderHistoricalCharts();
      await this.loadForecast();
    },

    setRoute(id) {
      this.route = id;
      window.location.hash = id;
      // Reflow charts si besoin
      this.$nextTick(() => {
        Object.values(this.charts).forEach(c => c?.resize?.());
      });
    },

    updateClock() {
      const d = new Date();
      this.now = d.toLocaleString('fr-FR', {
        weekday: 'short', day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit'
      });
    },

    welcomeMessage() {
      return {
        role: 'bot',
        content: "Bonjour ! Je suis l'assistant virtuel de la Tour Eiffel. Je peux répondre à vos questions sur les horaires, tarifs, accessibilité, restaurants ou l'histoire du monument — en français ou en anglais.",
      };
    },

    formatNumber: fmtNum,

    sentimentBadgeClass(s) {
      if (s === 'positive') return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      if (s === 'negative') return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
      return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
    },

    async fetchJSON(path, opts = {}) {
      try {
        const res = await fetch(API_BASE + path, opts);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return await res.json();
      } catch (e) {
        console.error('API error', path, e);
        return null;
      }
    },

    async loadDashboard() {
      this.dashboard = await this.fetchJSON('/api/dashboard/reviews');
    },

    async loadHistorical() {
      this.historical = await this.fetchJSON('/api/forecast/historical');
    },

    async loadLatestReviews() {
      const q = this.filterSentiment ? `?sentiment=${this.filterSentiment}&limit=15` : '?limit=15';
      const data = await this.fetchJSON('/api/dashboard/reviews/latest' + q);
      this.latestReviews = data?.reviews || [];
    },

    async loadPricing() {
      this.pricing = await this.fetchJSON('/api/dashboard/competitors');
    },

    async loadForecast() {
      this.forecast = await this.fetchJSON(`/api/forecast/day?date=${this.forecastDate}`);
      this.$nextTick(() => this.renderForecastChart());
    },

    async analyzeSentiment() {
      if (!this.sentimentInput.trim()) return;
      this.sentimentLoading = true;
      this.sentimentResult = null;
      this.sentimentResult = await this.fetchJSON('/api/sentiment/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: this.sentimentInput }),
      });
      this.sentimentLoading = false;
    },

    async sendChat() {
      const q = this.chatInput.trim();
      if (!q || this.chatLoading) return;
      this.messages.push({ role: 'user', content: q });
      this.chatInput = '';
      this.chatLoading = true;
      this.scrollChatToBottom();

      const res = await this.fetchJSON('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });

      this.chatLoading = false;
      if (res?.answer) {
        this.messages.push({
          role: 'bot',
          content: res.answer,
          confidence: res.confidence,
        });
      } else {
        this.messages.push({ role: 'bot', content: 'Désolé, une erreur est survenue.' });
      }
      this.scrollChatToBottom();
    },

    scrollChatToBottom() {
      this.$nextTick(() => {
        const el = document.getElementById('chatMessages');
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    // ============ CHARTS ============

    renderDashboardCharts() {
      if (!this.dashboard) return;
      this.destroyChart('chartMonthly');
      this.destroyChart('chartLang');
      this.destroyChart('chartTopics');
      this.destroyChart('chartSources');

      // Monthly evolution stacked area
      const months = this.dashboard.monthly_evolution || [];
      const labels = months.map(m => m.month);
      const ctx1 = document.getElementById('chartMonthly');
      if (ctx1) {
        this.charts.monthly = new Chart(ctx1, {
          type: 'line',
          data: {
            labels,
            datasets: [
              this.areaDataset('Positif', months.map(m => m.positive || 0), PALETTE.emerald),
              this.areaDataset('Neutre', months.map(m => m.neutral || 0), PALETTE.amber),
              this.areaDataset('Négatif', months.map(m => m.negative || 0), PALETTE.rose),
            ],
          },
          options: this.chartOptions({ stacked: true }),
        });
      }

      // Languages doughnut
      const ctx2 = document.getElementById('chartLang');
      if (ctx2) {
        const langs = this.dashboard.by_language;
        this.charts.lang = new Chart(ctx2, {
          type: 'doughnut',
          data: {
            labels: Object.keys(langs).map(l => l.toUpperCase()),
            datasets: [{
              data: Object.values(langs),
              backgroundColor: [PALETTE.gold, PALETTE.emerald, PALETTE.blue, PALETTE.purple, PALETTE.rose, PALETTE.amber],
              borderWidth: 0,
              hoverOffset: 8,
            }],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
              legend: { position: 'bottom', labels: { padding: 14, color: PALETTE.text, font: { size: 11 } } },
            },
          },
        });
      }

      // Topics bar
      const ctx3 = document.getElementById('chartTopics');
      if (ctx3) {
        const topics = this.dashboard.by_topic;
        const sorted = Object.entries(topics).sort((a, b) => b[1] - a[1]);
        this.charts.topics = new Chart(ctx3, {
          type: 'bar',
          data: {
            labels: sorted.map(([k]) => k),
            datasets: [{
              data: sorted.map(([, v]) => v),
              backgroundColor: 'rgba(212, 175, 55, 0.7)',
              borderColor: PALETTE.gold,
              borderWidth: 1,
              borderRadius: 6,
            }],
          },
          options: {
            ...this.chartOptions({ noStack: true, horizontal: true }),
            indexAxis: 'y',
            plugins: { legend: { display: false } },
          },
        });
      }

      // Sources pie
      const ctx4 = document.getElementById('chartSources');
      if (ctx4) {
        const sources = this.dashboard.by_source;
        this.charts.sources = new Chart(ctx4, {
          type: 'doughnut',
          data: {
            labels: Object.keys(sources),
            datasets: [{
              data: Object.values(sources),
              backgroundColor: [PALETTE.gold, PALETTE.blue, PALETTE.emerald],
              borderWidth: 0,
              hoverOffset: 8,
            }],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
              legend: { position: 'bottom', labels: { padding: 14, color: PALETTE.text, font: { size: 11 } } },
            },
          },
        });
      }
    },

    renderHistoricalCharts() {
      if (!this.historical) return;
      this.destroyChart('chartByDay');
      this.destroyChart('chartByMonth');

      const ctx5 = document.getElementById('chartByDay');
      if (ctx5) {
        const byDay = this.historical.by_day_of_week;
        this.charts.byDay = new Chart(ctx5, {
          type: 'bar',
          data: {
            labels: Object.keys(byDay),
            datasets: [{
              data: Object.values(byDay),
              backgroundColor: 'rgba(212, 175, 55, 0.75)',
              borderColor: PALETTE.gold,
              borderWidth: 1,
              borderRadius: 6,
            }],
          },
          options: { ...this.chartOptions({ noStack: true }), plugins: { legend: { display: false } } },
        });
      }

      const ctx6 = document.getElementById('chartByMonth');
      if (ctx6) {
        const byMonth = this.historical.by_month;
        this.charts.byMonth = new Chart(ctx6, {
          type: 'line',
          data: {
            labels: Object.keys(byMonth),
            datasets: [
              this.areaDataset('Visiteurs / heure', Object.values(byMonth), PALETTE.gold),
            ],
          },
          options: { ...this.chartOptions({ noStack: true }), plugins: { legend: { display: false } } },
        });
      }
    },

    renderForecastChart() {
      if (!this.forecast?.hourly) return;
      this.destroyChart('chartForecast');
      const ctx = document.getElementById('chartForecast');
      if (!ctx) return;

      const labels = this.forecast.hourly.map(h => h.hour + 'h');
      const values = this.forecast.hourly.map(h => h.visitors);

      this.charts.forecast = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Visiteurs prévus',
            data: values,
            borderColor: PALETTE.gold,
            backgroundColor: (context) => {
              const chart = context.chart;
              const { ctx, chartArea } = chart;
              if (!chartArea) return null;
              const grad = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
              grad.addColorStop(0, 'rgba(212, 175, 55, 0.4)');
              grad.addColorStop(1, 'rgba(212, 175, 55, 0)');
              return grad;
            },
            fill: true,
            tension: 0.35,
            pointBackgroundColor: PALETTE.gold,
            pointBorderColor: '#fff',
            pointBorderWidth: 1.5,
            pointRadius: 4,
            pointHoverRadius: 6,
            borderWidth: 2.5,
          }],
        },
        options: this.chartOptions({ noStack: true }),
      });
    },

    areaDataset(label, data, color) {
      return {
        label,
        data,
        borderColor: color,
        backgroundColor: color + '33',
        fill: true,
        tension: 0.3,
        pointRadius: 3,
        pointHoverRadius: 5,
        borderWidth: 2,
        pointBackgroundColor: color,
      };
    },

    chartOptions({ stacked = false, noStack = false, horizontal = false } = {}) {
      // Quand indexAxis='y' (barres horizontales), l'axe Y devient catégoriel
      // et l'axe X devient numérique : on inverse les formatters.
      const numFmt = function (val) { return fmtNum(val); };
      const catFmt = function (val) {
        // val est l'index, this.getLabelForValue(val) donne la string
        return this.getLabelForValue(val);
      };

      return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            backgroundColor: 'rgba(15, 26, 48, 0.95)',
            borderColor: 'rgba(212, 175, 55, 0.4)',
            borderWidth: 1,
            padding: 12,
            cornerRadius: 8,
            titleColor: '#fff',
            bodyColor: '#c6d2e3',
            titleFont: { weight: 600 },
          },
        },
        scales: {
          x: {
            grid: { color: PALETTE.grid, drawTicks: false },
            ticks: {
              color: PALETTE.text,
              font: { size: 11 },
              maxRotation: 45,
              autoSkip: true,
              maxTicksLimit: 14,
              callback: horizontal ? numFmt : catFmt,
            },
            stacked: stacked && !noStack,
          },
          y: {
            grid: { color: PALETTE.grid, drawTicks: false },
            ticks: {
              color: PALETTE.text,
              font: { size: 11 },
              callback: horizontal ? catFmt : numFmt,
            },
            stacked: stacked && !noStack,
            beginAtZero: true,
          },
        },
      };
    },

    destroyChart(canvasId) {
      const chart = Chart.getChart(canvasId);
      if (chart) chart.destroy();
    },
  };
}
