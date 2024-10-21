import setuptools
from setuptools import setup




setup(
    name="customer_analysis",
    version="0.3",
    description="Predicting the probability of chargeoff and then calculating the value at risk",
    license="MIT",
#     packages=setuptools.find_packages(include=["src/customer_analysis","src/customer_analysis.*"]),
    packages=setuptools.find_packages(where='src'),
    package_dir={
        '': 'src',
    },
    python_requires=">=3.7, <4",
    install_requires=[
        "markdown",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
    ],
    package_data={  # Optional
        "customer_analysis": ["final_model.pkl"],
    },
    zip_safe=False,
    test_suite="nose.collector",
    tests_require=["nose", "nose-cover3"],
    # scripts=["bin/funniest-joke"],
)