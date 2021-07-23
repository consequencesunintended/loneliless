<h2>LONELILESS</h2>
<a href="https://www.buymeacoffee.com/banterless" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
<p>&nbsp;</p>
<p id="yui_3_17_2_1_1569711346218_14531" class="">A Deep-Q Network playing a single player Pong game. Network done in Python (Tensorflow-gpu) with the single player Pong game implemented in C++ (Openframeworks) and both binded with Pybind11.</p>
<p class=""><img src="https://static1.squarespace.com/static/5d8c1173d980a856238b719f/t/5d8d36e5b01e692e31e2e6d0/1569535752308/Hnet-image+%281%29.gif?format=300w" alt="" width="160" height="160" /></p>
<h2>Instructions</h2>
<p>- Download the project from Github and unzip in a desired folder.&nbsp;</p>
<p>- Download Pybind11 from the Github page <a href="https://github.com/pybind/pybind11">https://github.com/pybind/pybind11</a> &nbsp;and copy the contents of the <strong>include</strong>, folder which should be only one folder called <strong>pybind11</strong>, to your Python default folder which in my case is <strong>C:\Python36_64\include</strong>.</p>
<p>- Download <a href="https://openframeworks.cc/download/">https://openframeworks.cc/download/</a> for Windows and unzip the file. Run <strong>projectGenerator.exe</strong> from the <strong>projectGenerator-vs</strong> folder in your unzipped files.</p>
<p>- Click on the magnifier in the Project path to change the path to the path you unzipped the Github project. Also change the Project name to the Github project name which is <strong>loneliless-master</strong></p>
<p>- Now the green <strong>Generate</strong> button should have changed to <strong>Update</strong>. Now click on Update to add OpenFrameworks engine to the downloaded project. You should be getting a message like the one below</p>
<p>- Click &ldquo;Open in IDE&rdquo;. (or you can manually do that by going to you project folder and clicking on <strong>loneliless.sln</strong>)</p>
<p>-In Visual Studio now you need to fix the linking path for Pybind11.</p>
<p>-Right click on the project and click properties</p>
<p>-Select C/C++</p>
<p>-Install pybind11 by running this command in command line: pip install pybind11 </p>
<p>-Navigate now to the Linker</p>
<p>-And add the python folder to the &ldquo;Additional Library Directories&rdquo;</p>
<p>-Click Ok and now you should be able to run the project.</p>
<h2>Dependencies</h2>
<p><a href="https://openframeworks.cc/" target="_blank" rel="noopener">OpenFrameworks</a></p>
<p><a href="https://github.com/pybind/pybind11" target="_blank" rel="noopener">Pybind11</a></p>
<p><a href="https://www.python.org/downloads/release/python-364/" target="_blank" rel="noopener">Python 3.x</a></p>
<p>Tensorflow 2.x</p>
<p>Numpy</p>
