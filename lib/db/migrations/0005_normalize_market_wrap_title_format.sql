UPDATE market_wrap SET title = replace(title, 'Markets Wrap: ', 'Market Wrap: ')
WHERE title LIKE 'Markets Wrap: %';

UPDATE market_wrap SET title = replace(title, 'Market Wrap — ', 'Market Wrap: ')
WHERE title LIKE 'Market Wrap — %';
