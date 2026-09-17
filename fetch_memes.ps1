<#
  Downloads the meme set this fork is tuned for into .\assets\, named after the pose each one belongs to.
  Every URL was checked frame by frame (not by its title). Run from the repo root:
      powershell -ExecutionPolicy Bypass -File fetch_memes.ps1
  A pose only fires if assets\ has a file (or folder) named after it; delete a file to switch its pose off.
  Pools: several files named <pose>~anything.gif are picked at random each time the pose fires.
#>
$ErrorActionPreference = "Stop"
$assets = Join-Path $PSScriptRoot "assets"
$memes = [ordered]@{
  "time_out.gif"        = "https://media.tenor.com/k2gx8ff-g3AAAAAC/stop-it-yanukovich.gif"             # Yanukovych "Astanavites"
  "talking_to_wall.gif" = "https://media.tenor.com/4LvAD8hD5tcAAAAC/charlie-day.gif"                    # Pepe Silvia, wide shot
  "hands_on_hips.gif"   = "https://media.tenor.com/LkzGJKbfBhEAAAAC/nu-da.gif"                          # "nu da, nu da..." (True Detective)
  "crashing_out.gif"    = "https://media.tenor.com/nuCH_1fvTXsAAAAd/laughing-hysterically-funny.gif"    # McAvoy, "what is going on"
  "nihuya.gif"          = "https://media.tenor.com/9V80o-wNni8AAAAC/nikuya.gif"                         # Tinkov
  "laugh.gif"           = "https://media1.tenor.com/m/QgTx6fv4IpAAAAAd/el-risitas-juan-joya-borja.gif"  # El Risitas
  "hello.gif"           = "https://media1.tenor.com/m/Tsob5aHiS3UAAAAd/hello-there.gif"                 # Kenobi
  "open_mouth.png"      = "https://i.kym-cdn.com/entries/icons/original/000/027/475/Screen_Shot_2018-10-25_at_11.02.15_AM.png"  # Surprised Pikachu
  # "shifty_eyes.gif"   = "https://media1.tenor.com/m/0i0ZxbRgVyAAAAAd/meonly.gif"                      # math lady (pose is twitchy; off by default)
}
foreach ($name in $memes.Keys) {
  $dest = Join-Path $assets $name
  Write-Host "-> $name"
  Invoke-WebRequest -Uri $memes[$name] -OutFile $dest -UseBasicParsing -Headers @{ "User-Agent" = "Mozilla/5.0" }
}
Write-Host "Done. Files sharing a pose name (time_out.jpeg + time_out.gif) form a random pool - delete the ones you do not want."
