rm -rf docpush
git clone --depth=1 git@github.com:facebookresearch/droidlet.git -b gh-pages docpush

cp -r build/html/* docpush/

pushd docpush/

git add * && git commit -m "document update"
git push origin gh-pages:gh-pages

popd
