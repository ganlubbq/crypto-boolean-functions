# CryptoBooleanFunctions

<h1>Overview</h2>
<p>The application is designed for computing main cryptographic characteristics of boolean functions. In use GPGPU model for speed up the
computations. The speed up over ordinary serial programms may be up to 60 times.<p>

<h2>Characteristics description</h2>
<p>
<ul>
  <li><code>disbalance</code> - computes value: abs(#{x: f(x) = 1} - {x: f(x) = 0}).</li>
  <li><code>degree</code> - computes algebraic degree of boolean function.</li>
  <li><code>nonlinearity</code> - computes nonlinearity as max distance to linear space of aphine functions.</li>
  <li><code>correlation_immunity</code> - computes correlation immunity of boolean functions.</li>
  <li><code>coefficient_of_error_propagation</code>, <code>sac</code> and etc. - set of functions to analysing how errors in input effects on output</li>
</ul>
</p>
<h2>Fast Dicrete Transforms</h2>
<p>
<ul>
<li><code>fft</code> - Fast Fourier Transform.
<li><code>fwt</code> - Fast Walsh Hadamard Transform.
<li><code>fmt</code> - Fast Mobius Transform.
</ul>
</p>
