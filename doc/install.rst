.. _install:

Installing
==========

There are different ways to install Tremelique:

.. tab-set::

    .. tab-item:: pip

        Using the `pip <https://pypi.org/project/pip/>`__ package manager:

        .. code:: bash

            python -m pip install tremelique

    .. tab-item:: conda/mamba

        Using the `conda package manager <https://conda.io/>`__ (or ``mamba``)
        that comes with the Anaconda/Miniconda distribution:

        .. code:: bash

            conda install tremelique --channel conda-forge

    .. tab-item:: Development version

        You can use ``pip`` to install the latest **unreleased** version from
        GitHub (**not recommended** in most situations):

        .. code:: bash

            python -m pip install --upgrade git+https://github.com/fatiando/tremelique

.. note::

    The commands above should be executed in a terminal. On Windows, use the
    ``cmd.exe`` or the "Anaconda Prompt" app if you're using Anaconda.

Which Python?
-------------

You'll need **Python >= 3.8**.
See :ref:`python-versions` if you require support for older versions.

.. _dependencies:

Dependencies
------------

The required dependencies should be installed automatically when you install
Tremelique using ``conda`` or ``pip``.

Required:

