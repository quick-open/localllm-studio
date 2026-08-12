; Inno Setup — LocalLLM Studio. Signed single-file installer, compiled in CI.
#define AppName "LocalLLM Studio"
#define AppVersion "1.0.2"

[Setup]
AppMutex=QuickOpen.LocalLLMStudio
AppId={{51A0F001-0018-4E5B-8C71-9B0E2F3A0018}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=QuickOpen (quickopen.ai)
AppPublisherURL=https://quickopen.ai/projects/localllm-studio
DefaultDirName={autopf}\LocalLLMStudio
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\LocalLLMStudio.exe
OutputDir=dist
OutputBaseFilename=LocalLLMStudio-Setup
SetupIconFile=..\localllm-studio.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
WizardImageFile=branding\wizard-large.bmp
WizardSmallImageFile=branding\wizard-small.bmp
AppCopyright=Apache-2.0. 100%% AI-built, published on QuickOpen (quickopen.ai).
VersionInfoCompany=QuickOpen
VersionInfoProductName=LocalLLM Studio
VersionInfoVersion=1.0.2.0
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel2=LocalLLM Studio is a 100%% AI-built, open-source offline tool, published on QuickOpen (quickopen.ai).%n%nThis will install it on your computer.
BeveledLabel=QuickOpen · quickopen.ai

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"
Name: "trustca"; Description: "Trust the QuickOpen Root CA (lets Windows verify QuickOpen signatures)"; GroupDescription: "Security:"; Flags: unchecked

[Files]
Source: "staging\LocalLLMStudio.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "staging\quickopen-root.crt"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "staging\README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme skipifsourcedoesntexist
Source: "staging\LICENSE"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\LocalLLM Studio"; Filename: "{app}\LocalLLMStudio.exe"; IconFilename: "{app}\LocalLLMStudio.exe"
Name: "{group}\Uninstall LocalLLM Studio"; Filename: "{uninstallexe}"
Name: "{autodesktop}\LocalLLM Studio"; Filename: "{app}\LocalLLMStudio.exe"; IconFilename: "{app}\LocalLLMStudio.exe"; Tasks: desktopicon

[Run]
Filename: "certutil.exe"; Parameters: "-addstore -user Root ""{app}\quickopen-root.crt"""; Tasks: trustca; Flags: runhidden; StatusMsg: "Trusting the QuickOpen Root CA..."
Filename: "{app}\LocalLLMStudio.exe"; Description: "Launch LocalLLM Studio now"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\LocalLLMStudio"

