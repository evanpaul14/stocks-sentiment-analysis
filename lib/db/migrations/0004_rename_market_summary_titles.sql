UPDATE market_wrap
SET title = replace(title, 'Market Summary', 'Market Wrap')
WHERE title LIKE '%Market Summary%';
