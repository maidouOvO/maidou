const { app, BrowserWindow } = require('electron')
const path = require('path')
const isDev = process.env.NODE_ENV === 'development'
const fs = require('fs')

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  })

  win.webContents.on('did-finish-load', () => {
    console.log('Window content loaded successfully')
    win.webContents.executeJavaScript(`
      console.log('DOM Content:', document.body.innerHTML);
      console.log('Loaded Assets:', Array.from(document.styleSheets).length, 'stylesheets');
    `)
  })

  win.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('Failed to load:', errorDescription, 'Code:', errorCode)
    setTimeout(() => {
      console.log('Retrying load...')
      if (isDev) {
        win.loadURL('http://localhost:5173')
      } else {
        win.loadFile(path.join(__dirname, '../dist/index.html'))
      }
    }, 1000)
  })

  if (isDev) {
    win.loadURL('http://localhost:5173')
    win.webContents.openDevTools()
  } else {
    const indexPath = path.join(__dirname, '../dist/index.html')
    console.log('Loading production build from:', indexPath)

    if (!fs.existsSync(indexPath)) {
      console.error('Error: index.html not found at', indexPath)
      console.log('Current directory contents:', fs.readdirSync(path.join(__dirname, '..')))
      app.quit()
      return
    }

    win.loadFile(indexPath)
    win.webContents.openDevTools()
  }
}

app.whenReady().then(() => {
  createWindow()
  console.log('App is packaged:', app.isPackaged)
  console.log('App path:', app.getAppPath())
  console.log('Current working directory:', process.cwd())
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})
