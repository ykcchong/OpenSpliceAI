<img alt="My Logo" class="logo header-image only-light align-center" src="logo/logo_black.png" style="width:90%">


<a class="reference external image-reference" href="https://img.shields.io/badge/License-GPLv3-yellow.svg"><img alt="https://img.shields.io/badge/License-GPLv3-yellow.svg" src="https://img.shields.io/badge/License-GPLv3-yellow.svg"></a>
<a class="reference external image-reference" href="https://img.shields.io/badge/version-v.0.0.1-blue"><img alt="https://img.shields.io/badge/version-v.0.0.1-blue" src="https://img.shields.io/badge/version-v.0.0.1-blue"></a>
<a class="reference external image-reference" href="https://pepy.tech/project/lifton"><img alt="https://static.pepy.tech/personalized-badge/lifton?period=total&amp;units=abbreviation&amp;left_color=grey&amp;right_color=blue&amp;left_text=PyPi%20downloads" src="https://static.pepy.tech/personalized-badge/lifton?period=total&amp;units=abbreviation&amp;left_color=grey&amp;right_color=blue&amp;left_text=PyPi%20downloads"></a>
<a class="reference external image-reference" href="https://github.com/Kuanhao-Chao/lifton/releases"><img alt="https://img.shields.io/github/downloads/Kuanhao-Chao/lifton/total.svg?style=social&amp;logo=github&amp;label=Download" src="https://img.shields.io/github/downloads/Kuanhao-Chao/lifton/total.svg?style=social&amp;logo=github&amp;label=Download"></a>
<a class="reference external image-reference" href="https://github.com/Kuanhao-Chao/spliceAI-toolkit/releases"><img alt="https://img.shields.io/badge/platform-macOS_/Linux-green.svg" src="https://img.shields.io/badge/platform-macOS_/Linux-green.svg"></a>
<a class="reference external image-reference" href="https://colab.research.google.com/github/Kuanhao-Chao/lifton/blob/main/notebook/lifton_example.ipynb"><img alt="https://colab.research.google.com/assets/colab-badge.svg" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
<div class="line-block">
<div class="line"><br></div>
</div>
<p>The SpliceAI-toolkit is a flexible framework designed for easy retraining of the SpliceAI model with new datasets. It comes with models pre-trained on various species, including humans (MANE database), mice, thale cress (Arabidopsis), honey bees, and zebrafish. Additionally, the SpliceAI-toolkit is capable of processing genetic variants in VCF format to predict their impact on splicing.</p>
<section id="why-spliceai-toolkit">
<h1>Why SpliceAI-toolkit❓<a class="headerlink" href="#why-spliceai-toolkit" title="Permalink to this heading">#</a></h1>
<ol class="arabic simple">
<li><p><strong>Easy-to-retrain framework</strong>: Transitioning from the outdated Python 2.7, along with older versions of TensorFlow and Keras, the SpliceAI-toolkit is built on Python 3.7 and leverages the powerful PyTorch library. This simplifies the retraining process significantly. Say goodbye to compatibility issues and hello to efficiency — retrain your models with just two simple commands.</p></li>
<li><p><strong>Pretrained on new dataset</strong>: SpliceAI is great, but SpliceAI-toolkit makes it even better! Pretrained with the latest MANE annotations (released in 2022), it ensures your research is powered by the most accurate and up-to-date genomic information available.</p></li>
<li><p><strong>Pretrained on various species</strong>:  Concerned that the SpliceAI model does not generalize to your study species because you are not studying humans? No problem! The SpliceAI-toolkit is released with models pretrained on various species, including human MANE, mouse, thale cress, honey bee, and zebrafish.</p></li>
<li><p><strong>Predict the impact of genetic variants on splicing</strong>: Similar to SpliceAI, the SpliceAI-toolkit can take genetic variants in VCF format and predict the impact of these variants on splicing with any of the pretrained models.</p></li>
</ol>
<p>SpliceAI-toolkit is open-source, free, and combines the ease of Python with the power of PyTorch for accurate splicing predictions.</p>
<div class="line-block">
<div class="line"><br></div>
</div>
</section>
<section id="who-is-it-for">
<h1>Who is it for❓<a class="headerlink" href="#who-is-it-for" title="Permalink to this heading">#</a></h1>
<ol class="arabic simple">
<li><p>If you want to study splicing in humans, just use the newly pretrained human SpliceAI-MANE! Better annotation, better results!</p></li>
<li><p>If you want to do splicing research in other species, the SpliceAI-toolkit has you covered! It comes with models pretrained on various species! And you can easily train your own SpliceAI with your own genome &amp; annotation data.</p></li>
<li><p>If you are interested in predicting the impact of genetic variants on splicing, SpliceAI-toolkit is the perfect tool for you!</p></li>
</ol>
<div class="line-block">
<div class="line"><br></div>
</div>
</section>
<section id="what-does-spliceai-toolkit-do">
<h1>What does SpliceAI-toolkit do❓<a class="headerlink" href="#what-does-spliceai-toolkit-do" title="Permalink to this heading">#</a></h1>
<ul class="simple">
<li><p>The spliceai-toolkit <code class="code docutils literal notranslate"><span class="pre">create-data</span></code> command takes a genome and annotation file as input and generates a dataset for training and testing your SpliceAI model.</p></li>
<li><p>The spliceai-toolkit <code class="code docutils literal notranslate"><span class="pre">train</span></code> command uses the created dataset to train your own SpliceAI model.</p></li>
<li><p>The spliceai-toolkit <code class="code docutils literal notranslate"><span class="pre">predict</span></code> command takes a random gene sequence and predicts the score of each position, determining whether it is a donor, acceptor, or neither.</p></li>
<li><p>The spliceai-toolkit <code class="code docutils literal notranslate"><span class="pre">variant</span></code> command takes a VCF file and predicts the impact of genetic variants on splicing.</p></li>
</ul>
<div class="line-block">
<div class="line"><br></div>
</div>
</section>
<section id="cite-us">
<h1>Cite us<a class="headerlink" href="#cite-us" title="Permalink to this heading">#</a></h1>
<p>Chao, Kua-Hao, Alan Mao, Anqi Liu, Mihaela Pertea, and Steven L. Salzberg. <i>"SpliceAI-toolkit"</i> <b>bioRxiv</b>.
<a href="https://doi.org/10.1093/bioinformatics/btaa1016" target="_blank"> <svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" x="0px" y="0px" viewBox="0 0 100 100" width="15" height="15" class="icon outbound"><path fill="currentColor" d="M18.8,85.1h56l0,0c2.2,0,4-1.8,4-4v-32h-8v28h-48v-48h28v-8h-32l0,0c-2.2,0-4,1.8-4,4v56C14.8,83.3,16.6,85.1,18.8,85.1z"></path> <polygon fill="currentColor" points="45.7,48.7 51.3,54.3 77.2,28.5 77.2,37.2 85.2,37.2 85.2,14.9 62.8,14.9 62.8,22.9 71.5,22.9"></polygon></svg> </a> </p>

<p>Jaganathan, K., Panagiotopoulou, S.K., McRae, J.F., Darbandi, S.F., Knowles, D., Li, Y.I., Kosmicki, J.A., Arbelaez, J., Cui, W., Schwartz, G.B. and Chow, E.D.<i>"Predicting splicing from primary sequence with deep learning"</i> <b>Cell</b>.
<a href="https://doi.org/10.1016/j.cell.2018.12.015" target="_blank"> <svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" x="0px" y="0px" viewBox="0 0 100 100" width="15" height="15" class="icon outbound"><path fill="currentColor" d="M18.8,85.1h56l0,0c2.2,0,4-1.8,4-4v-32h-8v28h-48v-48h28v-8h-32l0,0c-2.2,0-4,1.8-4,4v56C14.8,83.3,16.6,85.1,18.8,85.1z"></path> <polygon fill="currentColor" points="45.7,48.7 51.3,54.3 77.2,28.5 77.2,37.2 85.2,37.2 85.2,14.9 62.8,14.9 62.8,22.9 71.5,22.9"></polygon></svg> </a> </p><div class="line-block">
<div class="line"><br></div>
</div>
</section>
<section id="user-support">
<h1>User support<a class="headerlink" href="#user-support" title="Permalink to this heading">#</a></h1>
<p>Please go through the <a class="reference internal" href="#table-of-contents"><span class="std std-ref">documentation</span></a> below first. If you have questions about using the package, a bug report, or a feature request, please use the GitHub issue tracker here:</p>
<p><a class="reference external" href="https://github.com/Kuanhao-Chao/spliceAI-toolkit/issues">https://github.com/Kuanhao-Chao/spliceAI-toolkit/issues</a></p>
<div class="line-block">
<div class="line"><br></div>
</div>
</section>
<section id="key-contributors">
<h1>Key contributors<a class="headerlink" href="#key-contributors" title="Permalink to this heading">#</a></h1>
<p>SpliceAI-toolkit was designed and developed by <a class="reference external" href="https://khchao.com/">Kuan-Hao Chao</a>.  This documentation was written by <a class="reference external" href="https://khchao.com/">Kuan-Hao Chao</a>. The LiftOn logo was designed by <a class="reference external" href="https://khchao.com/">Kuan-Hao Chao</a>.</p>
<div class="line-block">
<div class="line"><br></div>
</div>
</section>
<section id="table-of-contents">
<span id="id3"></span><h1>Table of contents<a class="headerlink" href="#table-of-contents" title="Permalink to this heading">#</a></h1>
<div class="toctree-wrapper compound">
<ul>
<li class="toctree-l1"><a class="reference internal" href="content/installation.html">Installation</a><ul>
<li class="toctree-l2"><a class="reference internal" href="content/installation.html#system-requirements">System requirements</a></li>
<li class="toctree-l2"><a class="reference internal" href="content/installation.html#install-through-pip">Install through pip</a></li>
<li class="toctree-l2"><a class="reference internal" href="content/installation.html#install-through-conda">Install through conda</a></li>
<li class="toctree-l2"><a class="reference internal" href="content/installation.html#install-from-source">Install from source</a></li>
<li class="toctree-l2"><a class="reference internal" href="content/installation.html#check-spliceai-toolkit-installation">Check SpliceAI-toolkit installation</a></li>
<li class="toctree-l2"><a class="reference internal" href="content/installation.html#now-you-are-ready-to-go">Now, you are ready to go !</a></li>
</ul>
</li>
</ul>
</div>
<div class="toctree-wrapper compound">
<p class="caption" role="heading"><span class="caption-text">Examples</span></p>
<ul>
<li class="toctree-l1"><a class="reference internal" href="content/pretrained_models/index.html">Same species lift-over</a></li>
<li class="toctree-l1"><a class="reference internal" href="content/changelog.html">Changelog</a><ul>
<li class="toctree-l2"><a class="reference internal" href="content/changelog.html#v1-0-0">v1.0.0</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="content/license.html">License</a></li>
<li class="toctree-l1"><a class="reference internal" href="content/contact.html">Contact</a></li>
</ul>
</div>
<div class="line-block">
<div class="line"><br></div>
<div class="line"><br></div>
<div class="line"><br></div>
<div class="line"><br></div>
<div class="line"><br></div>
</div>

<br>
<br>
<br>

<img alt="My Logo" class="logo header-image only-light align-center" src="logo/jhu-logo-dark.png">

</section>
