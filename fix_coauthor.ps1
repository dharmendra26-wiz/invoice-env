# Run this in PowerShell inside c:\Users\dharm\invoice-env
# It amends all 3 today's commits to add Prachi as co-author

$coauthor = "Co-authored-by: Prachi01Yadav <archeyyadav111@gmail.com>"

# Rebase onto 11f08be, amending each commit message
git rebase --onto 11f08be 11f08be 574cb50ecf8a09ccf8b31a3b9ab446f283242ee4
$msg = git log -1 --format="%B"
git commit --amend -m "$msg`n`n$coauthor" --no-edit
