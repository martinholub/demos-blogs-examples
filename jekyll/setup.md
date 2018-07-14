# Setting up Jekyll on Linux  
First, install Ruby and dependencies.  

`sudo apt-get install ruby ruby-dev build-essential`  

Create path for gems in your home.  
```
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME=$HOME/gems' >> ~/.bashrc
echo 'export PATH=$HOME/gems/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```
Install jekyll and bundler.  
`gem install jekyll bundler`  
`bundle update jekyll`  
`gem update --system `  
Test with.  
`bundle exec jekyll --version`
