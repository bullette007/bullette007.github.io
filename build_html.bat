call bundle exec jekyll build
del _site\build_html.bat
rmdir C:\Users\meyjoh\repos\bullette007.github.io\jekyll /S /Q
Xcopy /E /I /Y _site C:\Users\meyjoh\repos\bullette007.github.io
cd C:\Users\meyjoh\repos\bullette007.github.io
git add *
git commit -m "Site update."
git push
pause