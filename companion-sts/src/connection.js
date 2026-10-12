// The websocket to sts's /control endpoint: connects, reconnects when it
// drops, keeps the latest on/off state, and sends toggle actions.
//
// Protocol (JSON text frames):
//   sts -> module  {"type": "status", "active", "translation_enabled", "voice_enabled", "voice_stopping", ...}
//                  {"type": "stage", "text", "ready"}       (start-up progress, "Listening" once ready)
//                  {"type": "error", "message"}             (an action it didn't accept)
//   module -> sts  {"action": "start" | "stop" | "translate_on" | "translate_off" | "voice_on" | "voice_off"}
//                  {"action": "get_status"}                 (send the status again)
// No authentication yet.

const WebSocket = require('ws')

const RECONNECT_MIN_MS = 1000
const RECONNECT_MAX_MS = 10000
const PING_EVERY_MS = 10000 // a connection that doesn't answer a ping by the next one is dropped

class StsConnection {
	/**
	 * @param {object} handlers
	 *   onState(state)            the state changed (see `state` below)
	 *   onConnection(connected, message)
	 *   log(level, text)
	 */
	constructor(handlers) {
		this.handlers = handlers
		this.url = null
		this.ws = null
		this.reconnectTimer = null
		this.pingTimer = null
		this.reconnectDelay = RECONNECT_MIN_MS
		this.alive = false
		this.closing = false
		this.state = {
			connected: false,
			caption: false, // "active" in sts
			translate: false,
			voice: false,
			voiceStopping: false, // Voice off was clicked: the sentence being spoken is finishing
			stage: '',
			ready: false,
		}
	}

	connect(host, port) {
		this.disconnect()
		this.closing = false
		this.url = `ws://${host}:${port}/control`
		this._open()
	}

	disconnect() {
		this.closing = true
		clearTimeout(this.reconnectTimer)
		clearInterval(this.pingTimer)
		this.reconnectTimer = null
		this.pingTimer = null
		if (this.ws) {
			this.ws.removeAllListeners()
			this.ws.on('error', () => {}) // a socket still connecting errors as it's torn down
			this.ws.terminate()
			this.ws = null
		}
		this._setConnected(false)
	}

	/** Sends one action; false if not connected. */
	send(action) {
		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
			this.handlers.log('warn', `not connected to sts: "${action}" not sent`)
			return false
		}
		this.ws.send(JSON.stringify({ action }))
		return true
	}

	_open() {
		this.handlers.onConnection(false, `connecting to ${this.url}`)
		const ws = new WebSocket(this.url, { handshakeTimeout: 5000 })
		this.ws = ws

		ws.on('open', () => {
			this.reconnectDelay = RECONNECT_MIN_MS
			this.alive = true
			this.handlers.log('info', `connected to ${this.url}`)
			this._setConnected(true)
			this.pingTimer = setInterval(() => {
				if (!this.alive) {
					this.handlers.log('warn', 'sts stopped answering; reconnecting')
					ws.terminate()
					return
				}
				this.alive = false
				ws.ping()
			}, PING_EVERY_MS)
		})
		ws.on('pong', () => {
			this.alive = true
		})
		ws.on('message', (data, isBinary) => {
			if (!isBinary) this._handle(data.toString())
		})
		ws.on('error', (err) => {
			this.handlers.log('debug', `websocket error: ${err.message}`)
		})
		ws.on('close', () => {
			clearInterval(this.pingTimer)
			this.pingTimer = null
			if (this.ws === ws) this.ws = null
			this._setConnected(false)
			if (!this.closing) this._scheduleReconnect()
		})
	}

	_scheduleReconnect() {
		clearTimeout(this.reconnectTimer)
		const delay = this.reconnectDelay
		this.reconnectDelay = Math.min(this.reconnectDelay * 2, RECONNECT_MAX_MS)
		this.handlers.onConnection(false, `can't reach sts at ${this.url}; retrying in ${Math.round(delay / 1000)} s`)
		this.reconnectTimer = setTimeout(() => this._open(), delay)
	}

	_handle(text) {
		let msg
		try {
			msg = JSON.parse(text)
		} catch {
			return
		}
		if (msg.type === 'status') {
			Object.assign(this.state, {
				caption: !!msg.active,
				translate: !!msg.translation_enabled,
				voice: !!msg.voice_enabled,
				voiceStopping: !!msg.voice_stopping,
			})
			this.handlers.onState(this.state)
		} else if (msg.type === 'stage') {
			Object.assign(this.state, { stage: msg.text || '', ready: !!msg.ready })
			this.handlers.onState(this.state)
		} else if (msg.type === 'error') {
			this.handlers.log('warn', `sts: ${msg.message}`)
		}
	}

	_setConnected(connected) {
		if (this.state.connected === connected) return
		this.state.connected = connected
		this.handlers.onConnection(connected, connected ? null : 'disconnected')
		this.handlers.onState(this.state)
	}
}

module.exports = { StsConnection }
