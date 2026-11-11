param([string]$Program, [string]$Directory)
$ErrorActionPreference = 'Stop'
$desktop = [Environment]::GetFolderPath('Desktop')
$shell = New-Object -ComObject WScript.Shell
$shortcutPath = Join-Path $desktop 'VoiceBankGen.lnk'
if (Test-Path -LiteralPath $shortcutPath) {
    $shortcutPath = Join-Path $desktop ('VoiceBankGen_' + [guid]::NewGuid().ToString('N').Substring(0, 8) + '.lnk')
}
$link = $shell.CreateShortcut($shortcutPath)
$link.TargetPath = $Program
$link.WorkingDirectory = $Directory
$link.IconLocation = "$Program,0"
$link.Save()
