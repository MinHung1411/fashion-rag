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
    selectedImageFile: null,
    selectedImagePreview: '',
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
        let guestId = localStorage.getItem('fashion_guest_id');
        if (!guestId) {
          guestId = 'guest_' + Math.random().toString(36).substring(2, 10);
          localStorage.setItem('fashion_guest_id', guestId);
        }

        const res = await fetch('/api/auth/quick-demo', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ guest_id: guestId })
        });
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
          // Fresh clean start! Always ensure activeConversationId is null on entry
          this.activeConversationId = null;
          this.activeConversationTitle = '';
          this.messages = [];
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

    async clearAllConversations() {
      if (!confirm('Bạn có chắc chắn muốn xóa toàn bộ lịch sử trò chuyện không?')) return;
      try {
        const res = await fetch('/api/conversations', {
          method: 'DELETE',
          headers: this.authHeaders
        });
        if (res.ok) {
          this.conversations = [];
          this.createNewConversation();
        }
      } catch (e) {
        console.error('Error clearing conversations:', e);
      }
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

    handleImageSelect(event) {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      this.selectedImageFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        this.selectedImagePreview = e.target.result;
      };
      reader.readAsDataURL(file);
    },

    removeSelectedImage() {
      this.selectedImageFile = null;
      this.selectedImagePreview = '';
      const inputEl = document.querySelector('input[type="file"]');
      if (inputEl) inputEl.value = '';
    },

    async sendMessage() {
      const query = this.inputQuery.trim();
      const hasImage = !!this.selectedImageFile;
      if ((!query && !hasImage) || this.isGenerating) return;

      if (!this.token) {
        this.showAuthModal = true;
        return;
      }

      // Ensure active conversation
      if (!this.activeConversationId) {
        try {
          const convTitle = query ? (query.slice(0, 36) + (query.length > 36 ? '...' : '')) : 'Tìm kiếm hình ảnh';
          const createRes = await fetch('/api/conversations', {
            method: 'POST',
            headers: this.authHeaders,
            body: JSON.stringify({ title: convTitle })
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
        content: query || 'Tìm kiếm sản phẩm tương tự bằng hình ảnh',
        image_url: this.selectedImagePreview || null,
        created_at: new Date().toISOString()
      };
      this.messages.push(tempUserMsg);

      const capturedImageFile = this.selectedImageFile;
      this.inputQuery = '';
      this.removeSelectedImage();
      this.isGenerating = true;
      this.$nextTick(() => this.scrollToBottom());

      // If user uploaded an image -> Visual Search endpoint
      if (hasImage && capturedImageFile) {
        try {
          const formData = new FormData();
          formData.append('file', capturedImageFile);
          if (query) formData.append('query', query);
          formData.append('top_k', this.settings.top_k);
          if (this.settings.gender_filter) formData.append('gender_filter', this.settings.gender_filter);
          if (this.settings.category_filter) formData.append('category_filter', this.settings.category_filter);

          const res = await fetch(`/api/conversations/${this.activeConversationId}/messages/image`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${this.token}` },
            body: formData
          });

          if (res.ok) {
            const asstMsg = await res.json();
            this.messages.push(asstMsg);
            const convIdx = this.conversations.findIndex(c => c.id === this.activeConversationId);
            if (convIdx !== -1) this.conversations[convIdx].last_message = asstMsg.content;
          } else {
            const err = await res.json().catch(() => ({}));
            this.messages.push({
              id: Date.now() + 1,
              conversation_id: this.activeConversationId,
              sender: 'assistant',
              content: `⚠️ Lỗi xử lý ảnh: ${err.detail || 'Không thể tìm kiếm ảnh lúc này'}`,
              created_at: new Date().toISOString()
            });
          }
        } catch (e) {
          console.error('Visual search error:', e);
          this.messages.push({
            id: Date.now() + 1,
            conversation_id: this.activeConversationId,
            sender: 'assistant',
            content: '⚠️ Có lỗi khi tải ảnh lên. Vui lòng thử lại.',
            created_at: new Date().toISOString()
          });
        } finally {
          this.isGenerating = false;
          this.$nextTick(() => this.scrollToBottom());
        }
        return;
      }

      // Priority 2: Real-time Streaming Response (SSE)
      const tempAsstId = Date.now() + 1;
      const asstMsgObj = {
        id: tempAsstId,
        conversation_id: this.activeConversationId,
        sender: 'assistant',
        content: '',
        retrieved_products: [],
        grounding_status: 'none',
        matched_terms: [],
        latency: null,
        is_outfit: false,
        created_at: new Date().toISOString()
      };
      this.messages.push(asstMsgObj);
      const asstRef = this.messages[this.messages.length - 1];

      try {
        const res = await fetch(`/api/conversations/${this.activeConversationId}/messages/stream`, {
          method: 'POST',
          headers: this.authHeaders,
          body: JSON.stringify({
            content: query,
            top_k: this.settings.top_k,
            gender_filter: this.settings.gender_filter || null,
            category_filter: this.settings.category_filter || null
          })
        });

        if (!res.ok) {
          throw new Error('Không thể kết nối luồng trả lời.');
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const event = JSON.parse(line.slice(6));
                if (event.type === 'retrieved') {
                  asstRef.retrieved_products = event.products || [];
                  asstRef.grounding_status = event.grounding_status;
                  asstRef.matched_terms = event.matched_terms || [];
                  asstRef.latency = event.latency;
                  asstRef.is_outfit = event.is_outfit || false;
                  if (event.title) {
                    this.activeConversationTitle = event.title;
                    const convIdx = this.conversations.findIndex(c => c.id === this.activeConversationId);
                    if (convIdx !== -1) this.conversations[convIdx].title = event.title;
                  }
                  this.$nextTick(() => this.scrollToBottom());
                } else if (event.type === 'token') {
                  asstRef.content += event.token;
                  this.$nextTick(() => this.scrollToBottom());
                } else if (event.type === 'done') {
                  asstRef.id = event.message_id || asstRef.id;
                  const convIdx = this.conversations.findIndex(c => c.id === this.activeConversationId);
                  if (convIdx !== -1) this.conversations[convIdx].last_message = asstRef.content;
                }
              } catch (parseErr) {
                // ignore keep reading
              }
            }
          }
        }
      } catch (err) {
        console.error('Streaming error, falling back:', err);
        if (!asstRef.content) {
          asstRef.content = '⚠️ Có lỗi xảy ra trong quá trình sinh câu trả lời. Vui lòng thử lại sau.';
        }
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
