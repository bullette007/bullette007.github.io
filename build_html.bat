call bundle exec jekyll build
del _site\build_html.bat
rmdir C:\Users\joh89642\repos\bullette007.github.io.pre-built\jekyll /S /Q
Xcopy /E /I /Y _site C:\Users\joh89642\repos\bullette007.github.io.pre-built
cd C:\Users\joh89642\repos\bullette007.github.io.pre-built
git add *
git commit -m "Site update."
git push
pause