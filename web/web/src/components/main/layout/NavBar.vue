<!-- components/Navbar.vue -->
<template>
  <nav class="navbar" :class="{ 'navbar-scrolled': isScrolled }">
    <div class="navbar-container">
      <!-- Logo区域 -->
      <div class="navbar-brand">
        <img src="/logo.png" alt="Logo" class="logo-image" />
        <!-- <div class="logo-placeholder">{{ title }}</div> -->
      </div>

      <!-- 电脑端导航菜单 -->
      <div class="navbar-menu">
        <ul class="navbar-nav">
          <li v-for="item in navItems" :key="item.id" class="nav-item">
            <a
              href="#"
              class="nav-link"
              :class="{ active: currentPage === item.page }"
              @click.prevent="changePage(item.page)"
            >
              {{ item.name }}
            </a>
          </li>
        </ul>
      </div>

      <!-- 右侧功能区 -->
      <div class="navbar-actions">
        <!-- 移动端汉堡菜单按钮 -->
        <button
          class="mobile-menu-toggle"
          :class="{ active: isMenuOpen }"
          @click="toggleMobileMenu"
          aria-label="导航菜单"
        >
          <span></span>
          <span></span>
          <span></span>
        </button>

        <!-- 用户信息 -->
        <div class="user-menu">
          <button class="user-toggle" @click="toggleUserMenu">
            <div class="user-avatar-placeholder">
              {{ user.name?.charAt(0) || 'U' }}
            </div>
            <span class="user-name">{{ user.name || '用户' }}</span>
          </button>

          <div class="user-dropdown" :class="{ active: isUserMenuOpen }">
            <div class="user-info">
              <div class="user-avatar-placeholder large">
                {{ user.name?.charAt(0) || 'U' }}
              </div>
              <div class="user-details">
                <h4>{{ user.name || '用户' }}</h4>
              </div>

              <div class="menu_button">
                <button @click="handleMenuClick('profile')">个人资料</button>
                <button @click="handleMenuClick('feedback')">反馈</button>
                <button @click="handleLogout">退出登录</button>
              </div>
            </div>
          </div>
        </div>
      </div>
      <!-- 移动端下拉菜单 -->
      <div class="mobile-menu" :class="{ active: isMenuOpen }">
        <ul class="mobile-nav">
          <li
            v-for="item in navItems"
            :key="item.id"
            class="mobile-nav-item"
          >
            <div
              class="mobile-nav-link"
              :class="{ active: currentPage === item.page }"
              @click="changePage(item.page)"
            >
              {{ item.name }}
        </div>
          </li>
        </ul>
      </div>
    </div>


  </nav>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import './styles/NavBar.css'
import type { NavItem } from '@/types/nav'

// Props
defineProps({
  brandName: {
    type: String,
    default: '我的应用',
  },
  navItems: {
    type: Array as () => NavItem[],
    default: () => [],
  },
  user: {
    type: Object,
    default: () => ({}),
  },
  currentPage: {
    type: String,
    default: 'home',
  },
})

// Emits
const emit = defineEmits(['page-change', 'menu-click', 'logout'])

// 响应式数据
const isScrolled = ref(false)
const isMenuOpen = ref(false)
const isUserMenuOpen = ref(false)

// 检测是否为移动端
const isMobile = ref(window.innerWidth <= 768)

// 更新屏幕尺寸
const updateScreenSize = () => {
  isMobile.value = window.innerWidth <= 768
}

// 方法
const changePage = (page: string) => {
  emit('page-change', page)
  closeMobileMenu()
}

const toggleMobileMenu = () => {
  isMenuOpen.value = !isMenuOpen.value
  // 移动端菜单打开时，关闭用户菜单
  if (isMenuOpen.value) {
    isUserMenuOpen.value = false
  }
}

const closeMobileMenu = () => {
  isMenuOpen.value = false
}

const toggleUserMenu = () => {
  // 移动端优先关闭移动菜单
  if (isMenuOpen.value && isMobile.value) {
    isMenuOpen.value = false
    setTimeout(() => {
      isUserMenuOpen.value = !isUserMenuOpen.value
    }, 300)
  } else {
    isUserMenuOpen.value = !isUserMenuOpen.value
  }
}

const handleMenuClick = (action: string) => {
  emit('menu-click', action)
  isUserMenuOpen.value = false
  closeMobileMenu()
}

const handleLogout = () => {
  emit('logout')
  isUserMenuOpen.value = false
  closeMobileMenu()
}

// 滚动监听
const handleScroll = () => {
  isScrolled.value = window.scrollY > 10
}

// 点击外部关闭菜单
const handleClickOutside = (event: MouseEvent) => {
  const target = event.target as HTMLElement
  const userMenu = target.closest('.user-menu')
  const mobileToggle = target.closest('.mobile-menu-toggle')
  const mobileMenu = target.closest('.mobile-menu')

  if (!userMenu) {
    isUserMenuOpen.value = false
  }

  if (!mobileToggle && !mobileMenu) {
    closeMobileMenu()
  }
}

// 键盘事件处理
const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Escape') {
    closeMobileMenu()
    isUserMenuOpen.value = false
  }
}

// 监听屏幕尺寸变化
watch(() => isMenuOpen.value, (newVal) => {
  if (newVal) {
    document.body.style.overflow = 'hidden'
  } else {
    document.body.style.overflow = ''
  }
})

// 生命周期
onMounted(() => {
  window.addEventListener('scroll', handleScroll)
  window.addEventListener('resize', updateScreenSize)
  document.addEventListener('click', handleClickOutside)
  document.addEventListener('keydown', handleKeydown)
  updateScreenSize() // 初始化
})

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll)
  window.removeEventListener('resize', updateScreenSize)
  document.removeEventListener('click', handleClickOutside)
  document.removeEventListener('keydown', handleKeydown)
  document.body.style.overflow = '' // 清理
})
</script>

<style scoped>
/* 保持原有样式不变 */
</style>
