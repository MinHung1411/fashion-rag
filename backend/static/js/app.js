/**
 * Fashion RAG — Frontend Application Controller (Alpine.js)
 * Author: Ho Minh Hung
 */

document.addEventListener('alpine:init', () => {
  Alpine.data('fashionApp', () => ({
    // Auth State
    token: localStorage.getItem('fashion_token') || '',
    currentUser: JSON.parse(localStorage.getItem('fashion_user') || 'null'),
    showAuthModal: false,
    authMode: 'login', // 'login' | 'register'
    authForm: {
      username: '',
      password: '',
      email: '',
      full_name: '',
      error: '',
      loading: false
    },

    // Chat & Conversation State
    conversations: [],
    activeConversationId: null,
    activeConversationTitle: '',
    messages: [],
    inputQuery: '',
    searchConvQuery: '',
    isGenerating: false,
    filterOptions: { genders: [], masterCategories: [] },

    // Tuning & Filters
    settings: {
      top_k: 5,
      gender_filter: '',
      category_filter: ''
    },

    // System Stats & Modals
    systemStats: {
      device: 'CPU',
      total_vectors: 44072,
      total_products: 44072,
      status: 'Operational'
    },
    examples: [],
    selectedProduct: null,
    showProductModal: false,
    showArchitectureModal: false,
    showSettingsModal: false,
    isDarkMode: true,
    sidebarOpen: true,

    async init() {
      // Check saved theme
      const savedTheme = localStorage.getItem('fashion_theme');
      if (savedTheme) {
        this.isDarkMode = savedTheme === 'dark';
      }

      await this.fetchStats();
      await this.fetchExamples();
      await this.fetchFilters();

      if (this.token && this.currentUser) {
        await this.loadConversations();
      } else {
        // Quick auto-login with demo for frictionless first visit
        await this.quickDemoLogin();
      }

      // Hotkey Ctrl+K / Cmd+K for new chat
      window.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
          e.preventDefault();
          this.createNewConversation();
        }
      });
    },

    get authHeaders() {
      return this.token ? { 'Authorization': `Bearer ${this.token}`, 'Content-Type': 'application/json' } : { 'Content-Type': 'application/json' };
    },

    get filteredConversations() {
      if (!this.searchConvQuery.trim()) return this.conversations;
      const q = this.searchConvQuery.toLowerCase();
      return this.conversations.filter(c => c.title.toLowerCase().includes(q));
    },

    toggleTheme() {
      this.isDarkMode = !this.isDarkMode;
      localStorage.setItem('fashion_theme', this.isDarkMode ? 'dark' : 'light');
    },

    // ── API Helpers ─────────────────────────────────────────────────────────────
    async fetchStats() {
      try {
        const res = await fetch('/api/stats');
        if (res.ok) {
          this.systemStats = await res.json();
        }
      } catch (e) {
        console.error('Error fetching stats:', e);
      }
    },

    async fetchExamples() {
      try {
        const res = await fetch('/api/rag/examples');
        if (res.ok) {
          this.examples = await res.json();
        }
      } catch (e) {
        console.error('Error fetching examples:', e);
      }
    },

    async fetchFilters() {
      try {
        const res = await fetch('/api/rag/filters');
        if (res.ok) {
          this.filterOptions = await res.json();
        }
      } catch (e) {
        console.error('Error fetching filters:', e);
      }
    },

    // ── Auth Methods ────────────────────────────────────────────────────────────
    async handleAuthSubmit() {
      this.authForm.error = '';
      this.authForm.loading = true;

      try {
        const url = this.authMode === 'login' ? '/api/auth/login' : '/api/auth/register';
        const body = this.authMode === 'login'
          ? { username: this.authForm.username, password: this.authForm.password }
          : {
              username: this.authForm.username,
              password: this.authForm.password,
              email: this.authForm.email,
              full_name: this.authForm.full_name
            };

        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Đã có lỗi xảy ra.');
        }

        this.token = data.access_token;
        this.currentUser = data.user;
        localStorage.setItem('fashion_token', this.token);
        localStorage.setItem('fashion_user', JSON.stringify(this.currentUser));

        this.showAuthModal = false;
        await this.loadConversations();
      } catch (err) {
        this.authForm.error = err.message;
      } finally {
        this.authForm.loading = false;
      }
    },

    async quickDemoLogin() {
      try {
        const res = await fetch('/api/auth/quick-demo', { method: 'POST' });
        if (res.ok) {
          const data = await res.json();
          this.token = data.access_token;
          this.currentUser = data.user;
          localStorage.setItem('fashion_token', this.token);
          localStorage.setItem('fashion_user', JSON.stringify(this.currentUser));
          this.showAuthModal = false;
          await this.loadConversations();
        }
      } catch (e) {
        console.error('Quick demo error:', e);
      }
    },

    logout() {
      this.token = '';
      this.currentUser = null;
      localStorage.removeItem('fashion_token');
      localStorage.removeItem('fashion_user');
      this.conversations = [];
      this.messages = [];
      this.activeConversationId = null;
      this.showAuthModal = true;
    },

    // ── Conversation & Messaging ───────────────────────────────────────────────
    async loadConversations() {
      if (!this.token) return;
      try {
        const res = await fetch('/api/conversations', { headers: this.authHeaders });
        if (res.ok) {
          this.conversations = await res.json();
          // Fresh clean start! Do not auto-select old conversations on load.
        }
      } catch (e) {
        console.error('Error loading conversations:', e);
      }
    },

    selectConversation(convId) {
      this.activeConversationId = convId;
      const conv = this.conversations.find(c => c.id === convId);
      if (conv) this.activeConversationTitle = conv.title;

      fetch(`/api/conversations/${convId}`, { headers: this.authHeaders })
        .then(res => res.json())
        .then(data => {
          this.messages = data.messages || [];
          this.activeConversationTitle = data.title;
          this.$nextTick(() => this.scrollToBottom());
        })
        .catch(e => console.error('Error fetching conversation details:', e));
    },

    createNewConversation() {
      this.activeConversationId = null;
      this.activeConversationTitle = '';
      this.messages = [];
      this.inputQuery = '';
      this.$nextTick(() => {
        const inputEl = document.querySelector('textarea');
        if (inputEl) inputEl.focus();
      });
    },

    async deleteConversation(convId, event) {
      if (event) event.stopPropagation();
      if (!confirm('Bạn có chắc chắn muốn xóa cuộc trò chuyện này?')) return;

      try {
        const res = await fetch(`/api/conversations/${convId}`, {
          method: 'DELETE',
          headers: this.authHeaders
        });
        if (res.ok) {
          this.conversations = this.conversations.filter(c => c.id !== convId);
          if (this.activeConversationId === convId) {
            if (this.conversations.length > 0) {
              await this.selectConversation(this.conversations[0].id);
            } else {
              this.activeConversationId = null;
              this.messages = [];
            }
          }
        }
      } catch (e) {
        console.error('Error deleting conversation:', e);
      }
    },

    async renameConversation(convId) {
      const current = this.conversations.find(c => c.id === convId);
      const newTitle = prompt('Nhập tiêu đề mới cho cuộc trò chuyện:', current ? current.title : '');
      if (!newTitle || !newTitle.trim()) return;

      try {
        const res = await fetch(`/api/conversations/${convId}`, {
          method: 'PATCH',
          headers: this.authHeaders,
          body: JSON.stringify({ title: newTitle.trim() })
        });
        if (res.ok) {
          const updated = await res.json();
          const idx = this.conversations.findIndex(c => c.id === convId);
          if (idx !== -1) this.conversations[idx].title = updated.title;
          if (this.activeConversationId === convId) this.activeConversationTitle = updated.title;
        }
      } catch (e) {
        console.error('Error renaming conversation:', e);
      }
    },

    useExample(query) {
      this.inputQuery = query;
      this.sendMessage();
    },

    async sendMessage() {
      const query = this.inputQuery.trim();
      if (!query || this.isGenerating) return;

      if (!this.token) {
        this.showAuthModal = true;
        return;
      }

      // Ensure active conversation
      if (!this.activeConversationId) {
        try {
          const createRes = await fetch('/api/conversations', {
            method: 'POST',
            headers: this.authHeaders,
            body: JSON.stringify({ title: query.slice(0, 36) + (query.length > 36 ? '...' : '') })
          });
          if (createRes.ok) {
            const newConv = await createRes.json();
            this.conversations.unshift(newConv);
            this.activeConversationId = newConv.id;
            this.activeConversationTitle = newConv.title;
          }
        } catch (e) {
          console.error('Error creating conversation:', e);
        }
      }

      // Optimistic user message
      const tempUserMsg = {
        id: Date.now(),
        conversation_id: this.activeConversationId,
        sender: 'user',
        content: query,
        created_at: new Date().toISOString()
      };
      this.messages.push(tempUserMsg);
      this.inputQuery = '';
      this.isGenerating = true;
      this.$nextTick(() => this.scrollToBottom());

      try {
        const res = await fetch(`/api/conversations/${this.activeConversationId}/messages`, {
          method: 'POST',
          headers: this.authHeaders,
          body: JSON.stringify({
            content: query,
            top_k: this.settings.top_k,
            gender_filter: this.settings.gender_filter || null,
            category_filter: this.settings.category_filter || null
          })
        });

        if (res.ok) {
          const asstMsg = await res.json();
          this.messages.push(asstMsg);
          
          // Update conversation in sidebar list
          const convIdx = this.conversations.findIndex(c => c.id === this.activeConversationId);
          if (convIdx !== -1) {
            this.conversations[convIdx].last_message = asstMsg.content;
            if (this.conversations[convIdx].title === 'Cuộc trò chuyện mới') {
              this.conversations[convIdx].title = query.slice(0, 36) + (query.length > 36 ? '...' : '');
              this.activeConversationTitle = this.conversations[convIdx].title;
            }
          }
        } else {
          const err = await res.json();
          alert('Lỗi: ' + (err.detail || 'Không thể gửi câu hỏi'));
        }
      } catch (e) {
        console.error('Error sending message:', e);
      } finally {
        this.isGenerating = false;
        this.$nextTick(() => this.scrollToBottom());
      }
    },

    scrollToBottom() {
      const container = document.getElementById('chat-scroll-container');
      if (container) {
        container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
      }
    },

    // ── Product Modal ───────────────────────────────────────────────────────────
    openProductModal(product) {
      this.selectedProduct = product;
      this.showProductModal = true;
    },

    closeProductModal() {
      this.showProductModal = false;
      this.selectedProduct = null;
    },

    // ── Export Chat ─────────────────────────────────────────────────────────────
    exportConversation(format = 'markdown') {
      if (this.messages.length === 0) return;

      let content = '';
      const title = this.activeConversationTitle || 'Fashion_RAG_Chat';

      if (format === 'markdown') {
        content += `# ${title}\n\n`;
        content += `*Ngày xuất: ${new Date().toLocaleString()}*\n\n---\n\n`;
        this.messages.forEach(m => {
          if (m.sender === 'user') {
            content += `### 👤 Người dùng:\n${m.content}\n\n`;
          } else {
            content += `### 🛍️ Fashion RAG Assistant:\n${m.content}\n\n`;
            if (m.grounding_status) {
              content += `> **Grounding:** ${m.grounding_status.toUpperCase()} | Độ trễ: ${m.latency?.total_ms || 0}ms\n\n`;
            }
            if (m.retrieved_products?.length) {
              content += `**Sản phẩm được gợi ý:**\n`;
              m.retrieved_products.forEach((p, i) => {
                content += `${i + 1}. **${p.productDisplayName}** (${p.baseColour}, ${p.gender}) — Điểm tương đồng: ${(p.score * 100).toFixed(1)}%\n`;
              });
              content += `\n`;
            }
          }
          content += `---\n\n`;
        });
      } else {
        content = JSON.stringify({
          title,
          exported_at: new Date().toISOString(),
          messages: this.messages
        }, null, 2);
      }

      const blob = new Blob([content], { type: format === 'markdown' ? 'text/markdown' : 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${title.replace(/\s+/g, '_')}.${format === 'markdown' ? 'md' : 'json'}`;
      a.click();
      URL.revokeObjectURL(url);
    }
  }));
});
