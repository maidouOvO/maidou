; PS绘本文本处理工具安装程序脚本
; 使用NSIS (Nullsoft Scriptable Install System) 创建Windows安装包

; 基本设置
!define APPNAME "PS绘本文本处理工具"
!define COMPANYNAME "绘本工具"
!define DESCRIPTION "自动处理绘本文本的工具"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0

; 包含现代UI
!include "MUI2.nsh"

; 一般设置
Name "${APPNAME}"
OutFile "PS绘本文本处理工具_安装包.exe"
InstallDir "$PROGRAMFILES\${APPNAME}"
InstallDirRegKey HKCU "Software\${APPNAME}" ""

; 请求应用程序权限
RequestExecutionLevel admin

; 界面设置
!define MUI_ABORTWARNING
!define MUI_ICON "icon.ico"
!define MUI_UNICON "icon.ico"

; 安装页面
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; 卸载页面
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; 语言设置
!insertmacro MUI_LANGUAGE "SimpChinese"

; 安装部分
Section "安装" SecInstall
  SetOutPath "$INSTDIR"
  
  ; 添加文件
  File "dist\ps_text_remover.exe"
  File "icon.ico"
  File "README.txt"
  
  ; 创建卸载程序
  WriteUninstaller "$INSTDIR\卸载.exe"
  
  ; 创建开始菜单快捷方式
  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\ps_text_remover.exe" "" "$INSTDIR\icon.ico"
  CreateShortCut "$SMPROGRAMS\${APPNAME}\卸载.lnk" "$INSTDIR\卸载.exe" "" "$INSTDIR\icon.ico"
  
  ; 创建桌面快捷方式
  CreateShortCut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\ps_text_remover.exe" "" "$INSTDIR\icon.ico"
  
  ; 写入注册表信息
  WriteRegStr HKCU "Software\${APPNAME}" "" $INSTDIR
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayName" "${APPNAME}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "UninstallString" "$\"$INSTDIR\卸载.exe$\""
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayIcon" "$\"$INSTDIR\icon.ico$\""
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "Publisher" "${COMPANYNAME}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}"
SectionEnd

; 卸载部分
Section "卸载"
  ; 删除文件
  Delete "$INSTDIR\ps_text_remover.exe"
  Delete "$INSTDIR\icon.ico"
  Delete "$INSTDIR\README.txt"
  Delete "$INSTDIR\卸载.exe"
  
  ; 删除快捷方式
  Delete "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk"
  Delete "$SMPROGRAMS\${APPNAME}\卸载.lnk"
  Delete "$DESKTOP\${APPNAME}.lnk"
  
  ; 删除目录
  RMDir "$SMPROGRAMS\${APPNAME}"
  RMDir "$INSTDIR"
  
  ; 删除注册表项
  DeleteRegKey HKCU "Software\${APPNAME}"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
SectionEnd
