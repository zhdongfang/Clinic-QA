<template>
    <div id="app">
      <ChatRobot />
    </div>
  </template>

<script>
import ChatRobot from './components/ChatRobot.vue'

export default {
  name: 'App',
  components: {
    ChatRobot
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  /* margin-top: 60px; */
}
</style>